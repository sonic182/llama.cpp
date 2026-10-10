use std::ffi::{CStr, CString, c_char, c_int};
use std::io::{self, Write};
use std::ptr;

use llama::sys;
use llama_quantize_core::args::{
    Console, Exit, Options, Target, parse_options, resolve_target, same_file, usage,
};
use llama_quantize_core::imatrix::{self, Data};
use llama_quantize_core::kv::{Override, Value};

use crate::imatrix::load;

unsafe extern "C" {
    fn llama_build_number() -> c_int;
    fn llama_commit() -> *const c_char;
    fn llama_compiler() -> *const c_char;
    fn llama_build_target() -> *const c_char;
}

fn type_names() -> Vec<Option<Vec<u8>>> {
    (0..sys::ggml_type::GGML_TYPE_COUNT)
        .map(|ty| {
            let name = unsafe { sys::ggml_type_name(ty) };
            (!name.is_null()).then(|| unsafe { CStr::from_ptr(name) }.to_bytes().to_vec())
        })
        .collect()
}

fn exit_with(code: i32, out: &mut dyn Write) -> ! {
    let _ = out.flush();
    std::process::exit(code)
}

fn write_quoted(w: &mut dyn Write, prefix: &str, value: &[u8], suffix: &str) {
    let _ = w.write_all(prefix.as_bytes());
    let _ = w.write_all(value);
    let _ = w.write_all(suffix.as_bytes());
}

struct Prepared {
    data: Data,
    datasets: Vec<Vec<u8>>,
    m_last_call: i32,
}

fn prepare_imatrix(o: &Options, out: &mut dyn Write, err: &mut dyn Write) -> Prepared {
    let mut prepared = Prepared {
        data: Data::new(),
        datasets: Vec::new(),
        m_last_call: -1,
    };
    if !o.imatrix_file.is_empty() {
        let file = &o.imatrix_file;
        let Some(loaded) = load(file, &mut |msg| {
            let _ = err.write_all(msg);
        }) else {
            write_quoted(
                err,
                "load_imatrix: failed to load imatrix from '",
                file,
                "'\n",
            );
            exit_with(1, out);
        };
        if !loaded.is_legacy && !loaded.has_metadata {
            write_quoted(
                err,
                "load_imatrix: missing imatrix metadata in file ",
                file,
                "\n",
            );
            exit_with(1, out);
        }
        let trace = std::env::var_os("LLAMA_TRACE").is_some();
        prepared.data = imatrix::normalize(&loaded, trace, out);
        prepared.datasets = loaded.datasets;
        if let Some((first, rest)) = prepared.datasets.split_first() {
            write_quoted(out, "load_imatrix: imatrix datasets=['", first, "'");
            for d in rest {
                write_quoted(out, ", '", d, "'");
            }
            let _ = out.write_all(b"]\n");
        }
        let _ = write!(
            out,
            "load_imatrix: loaded {} importance matrix entries from ",
            prepared.data.len()
        );
        let _ = out.write_all(file);
        let _ = writeln!(out, " computed on {} chunks", loaded.chunk_count);
        prepared.m_last_call = loaded.chunk_count;
    }
    if prepared.data.is_empty() {
        return prepared;
    }
    imatrix::filter(&mut prepared.data, &o.included_weights, &o.excluded_weights);
    if !prepared.data.is_empty() {
        let _ = writeln!(
            out,
            "prepare_imatrix: have {} importance matrix entries",
            prepared.data.len()
        );
    }
    prepared
}

fn copy_c(dst: &mut [c_char], src: &[u8]) {
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = s as c_char;
    }
    dst[src.len().min(dst.len() - 1)] = 0;
}

fn to_c_override(o: &Override) -> sys::llama_model_kv_override {
    let mut kvo: sys::llama_model_kv_override = unsafe { std::mem::zeroed() };
    copy_c(&mut kvo.key, &o.key);
    match &o.value {
        Value::Int(v) => {
            kvo.tag = sys::llama_model_kv_override_type::LLAMA_KV_OVERRIDE_TYPE_INT;
            kvo.__bindgen_anon_1.val_i64 = *v;
        }
        Value::Float(v) => {
            kvo.tag = sys::llama_model_kv_override_type::LLAMA_KV_OVERRIDE_TYPE_FLOAT;
            kvo.__bindgen_anon_1.val_f64 = *v;
        }
        Value::Bool(v) => {
            kvo.tag = sys::llama_model_kv_override_type::LLAMA_KV_OVERRIDE_TYPE_BOOL;
            kvo.__bindgen_anon_1.val_bool = *v;
        }
        Value::Str(v) => {
            kvo.tag = sys::llama_model_kv_override_type::LLAMA_KV_OVERRIDE_TYPE_STR;
            copy_c(unsafe { &mut kvo.__bindgen_anon_1.val_str }, v);
        }
    }
    kvo
}

fn cstring(bytes: &[u8]) -> CString {
    CString::new(bytes).unwrap_or_default()
}

pub fn run(argv: &[Vec<u8>]) -> i32 {
    unsafe { libc::setlocale(libc::LC_NUMERIC, c"C".as_ptr()) };

    let mut stdout = io::stdout().lock();
    let mut stderr = io::stderr();
    let out: &mut dyn Write = &mut stdout;
    let err: &mut dyn Write = &mut stderr;
    let argv0 = argv.first().cloned().unwrap_or_default();

    let names = type_names();
    let parsed = {
        let mut console = Console {
            out: &mut *out,
            err: &mut *err,
        };
        parse_options(argv, &names, &mut console)
    };
    let (o, idx) = match parsed {
        Ok(v) => v,
        Err(Exit::Usage) => {
            usage(&argv0, out);
            exit_with(1, out);
        }
        Err(Exit::Code(code)) => return code,
    };

    let prepared = prepare_imatrix(&o, out, err);

    let mut params = unsafe { sys::llama_model_quantize_default_params() };
    if o.leave_output_tensor {
        params.quantize_output_tensor = false;
    }
    if let Some(ty) = o.output_tensor_type {
        params.output_tensor_type = ty as sys::ggml_type::Type;
    }
    if let Some(ty) = o.token_embedding_type {
        params.token_embedding_type = ty as sys::ggml_type::Type;
    }
    params.dry_run = o.dry_run;
    params.allow_requantize = o.allow_requantize;
    params.pure_ = o.pure;
    params.keep_split = o.keep_split;
    if let Some(size) = o.max_buf_size {
        params.max_buf_size = size;
    }

    let names_c: Vec<CString> = prepared.data.keys().map(|n| cstring(n)).collect();
    let mut i_data: Vec<sys::llama_model_imatrix_data> = Vec::new();
    let mut kv_overrides = o.kv_overrides.clone();
    if !prepared.data.is_empty() {
        i_data.extend(prepared.data.values().zip(&names_c).map(|(v, name)| {
            sys::llama_model_imatrix_data {
                name: name.as_ptr(),
                data: v.as_ptr(),
                size: v.len(),
            }
        }));
        i_data.push(sys::llama_model_imatrix_data {
            name: ptr::null(),
            data: ptr::null(),
            size: 0,
        });
        params.imatrix = i_data.as_ptr();
        kv_overrides.push(Override::str("quantize.imatrix.file", &o.imatrix_file));
        if let Some(first) = prepared.datasets.first() {
            kv_overrides.push(Override::str("quantize.imatrix.dataset", first));
        }
        kv_overrides.push(Override::int(
            "quantize.imatrix.entries_count",
            prepared.data.len() as i64,
        ));
        if prepared.m_last_call > 0 {
            kv_overrides.push(Override::int(
                "quantize.imatrix.chunks_count",
                i64::from(prepared.m_last_call),
            ));
        }
    }
    let mut kv_c: Vec<sys::llama_model_kv_override> =
        kv_overrides.iter().map(to_c_override).collect();
    if !kv_c.is_empty() {
        kv_c.push(unsafe { std::mem::zeroed() });
        params.kv_overrides = kv_c.as_ptr();
    }
    let patterns: Vec<CString> = o.tensor_types.iter().map(|(n, _)| cstring(n)).collect();
    let mut t_override: Vec<sys::llama_model_tensor_override> = Vec::new();
    if !o.tensor_types.is_empty() {
        t_override.extend(o.tensor_types.iter().zip(&patterns).map(|((_, ty), p)| {
            sys::llama_model_tensor_override {
                pattern: p.as_ptr(),
                type_: *ty as sys::ggml_type::Type,
            }
        }));
        t_override.push(sys::llama_model_tensor_override {
            pattern: ptr::null(),
            type_: sys::ggml_type::GGML_TYPE_COUNT,
        });
        params.tt_overrides = t_override.as_ptr();
    }
    let mut prune_layers = o.prune_layers.clone();
    if !prune_layers.is_empty() {
        prune_layers.push(-1);
        params.prune_layers = prune_layers.as_ptr();
    }

    unsafe { sys::llama_backend_init() };

    let Target {
        input,
        output,
        ftype,
        nthread,
    } = match resolve_target(argv, idx, &o, err) {
        Ok(t) => t,
        Err(Exit::Code(code)) => return code,
        Err(Exit::Usage) => {
            usage(&argv0, out);
            exit_with(1, out);
        }
    };
    params.ftype = ftype.ftype as sys::llama_ftype::Type;
    if ftype.name == "COPY" {
        params.only_copy = true;
    }
    if let Some(n) = nthread {
        params.nthread = n;
    }

    if !o.dry_run && same_file(&input, &output) {
        write_quoted(
            err,
            "llama_quantize: error: input and output files are the same: '",
            &input,
            "'\n",
        );
        return 1;
    }

    print_build_info(err);

    if o.dry_run {
        write_quoted(
            err,
            "llama_quantize: calculating quantization size for '",
            &input,
            "' as ",
        );
    } else {
        write_quoted(err, "llama_quantize: quantizing '", &input, "' to '");
        write_quoted(err, "", &output, "' as ");
    }
    let _ = err.write_all(ftype.name.as_bytes());
    if params.nthread > 0 {
        let _ = write!(err, " using {} threads", params.nthread);
    }
    let _ = err.write_all(b"\n");

    let t_main_start_us = unsafe { sys::llama_time_us() };
    let input_c = cstring(&input);
    let output_c = cstring(&output);
    let t_start_us = unsafe { sys::llama_time_us() };
    if unsafe { sys::llama_model_quantize(input_c.as_ptr(), output_c.as_ptr(), &params) } != 0 {
        write_quoted(
            err,
            "llama_quantize: failed to quantize model from '",
            &input,
            "'\n",
        );
        return 1;
    }
    let t_quantize_us = unsafe { sys::llama_time_us() } - t_start_us;
    let t_main_end_us = unsafe { sys::llama_time_us() };

    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "llama_quantize: quantize time = {:8.2} ms",
        t_quantize_us as f64 / 1000.0
    );
    let _ = writeln!(
        out,
        "llama_quantize:    total time = {:8.2} ms",
        (t_main_end_us - t_main_start_us) as f64 / 1000.0
    );
    let _ = out.flush();

    unsafe { sys::llama_backend_free() };
    0
}

fn print_build_info(err: &mut dyn Write) {
    let text = |p: *const c_char| unsafe { CStr::from_ptr(p) }.to_bytes().to_vec();
    let version = text(unsafe { sys::llama_version() });
    let _ = err.write_all(b"version: ");
    let _ = err.write_all(&version);
    let _ = write!(err, " (build {}, commit ", unsafe { llama_build_number() });
    let _ = err.write_all(&text(unsafe { llama_commit() }));
    let _ = err.write_all(b")\nbuilt with ");
    let _ = err.write_all(&text(unsafe { llama_compiler() }));
    let _ = err.write_all(b" for ");
    let _ = err.write_all(&text(unsafe { llama_build_target() }));
    let _ = err.write_all(b"\n");
}
