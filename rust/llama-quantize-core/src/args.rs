use std::io::Write;
use std::path::Path;

use crate::cnum::{atoi, c_str, is_c_space, stoi, strerror};
use crate::ftype::{QUANT_OPTIONS, QuantOption, striequals, try_parse_ftype};
use crate::kv::{self, Override};

pub struct Console<'a> {
    pub out: &'a mut dyn Write,
    pub err: &'a mut dyn Write,
}

#[derive(Debug, PartialEq)]
pub enum Exit {
    Usage,
    Code(i32),
}

#[derive(Debug, Default)]
pub struct Options {
    pub leave_output_tensor: bool,
    pub output_tensor_type: Option<i32>,
    pub token_embedding_type: Option<i32>,
    pub tensor_types: Vec<(Vec<u8>, i32)>,
    pub prune_layers: Vec<i32>,
    pub kv_overrides: Vec<Override>,
    pub dry_run: bool,
    pub allow_requantize: bool,
    pub pure: bool,
    pub imatrix_file: Vec<u8>,
    pub included_weights: Vec<Vec<u8>>,
    pub excluded_weights: Vec<Vec<u8>>,
    pub keep_split: bool,
    pub max_buf_size: Option<usize>,
}

#[derive(Debug, PartialEq)]
pub struct Target {
    pub input: Vec<u8>,
    pub output: Vec<u8>,
    pub ftype: &'static QuantOption,
    pub nthread: Option<i32>,
}

impl Target {
    pub fn only_copy(&self) -> bool {
        self.ftype.name == "COPY"
    }
}

pub fn usage(argv0: &[u8], out: &mut dyn Write) {
    let _ = out.write_all(b"usage: ");
    let _ = out.write_all(argv0);
    let _ = out.write_all(USAGE.as_bytes());
    for it in QUANT_OPTIONS {
        if it.name != "COPY" {
            let _ = write!(out, "  {:2}  or  ", it.ftype);
        } else {
            let _ = write!(out, "          ");
        }
        let _ = writeln!(out, "{:<7} : {}", it.name, it.desc);
    }
}

const USAGE: &str = " [--help] [--allow-requantize] [--leave-output-tensor] [--pure] [--imatrix] [--include-weights]
       [--exclude-weights] [--output-tensor-type] [--token-embedding-type] [--tensor-type] [--tensor-type-file]
       [--prune-layers] [--keep-split] [--override-kv] [--dry-run] [--max-buffer-size]
       model-f32.gguf [model-quant.gguf] type [nthreads]

  --allow-requantize
                                      allow requantizing tensors that have already been quantized
                                      WARNING: this can severely reduce quality compared to quantizing
                                               from 16bit or 32bit!
  --leave-output-tensor
                                      leave output.weight un(re)quantized
                                      increases model size but may also increase quality, especially when requantizing
  --pure
                                      disable k-quant mixtures and quantize all tensors to the same type
  --imatrix file_name
                                      use data in file_name as importance matrix for quant optimizations
  --include-weights tensor_name
                                      use importance matrix for this/these tensor(s)
  --exclude-weights tensor_name
                                      do not use importance matrix for this/these tensor(s)
  --output-tensor-type ggml_type
                                      use this ggml_type for the output.weight tensor
  --token-embedding-type ggml_type
                                      use this ggml_type for the token embeddings tensor
  --tensor-type tensor_name=ggml_type
                                      quantize this tensor to this ggml_type
                                      this is an advanced option to selectively quantize tensors. may be specified multiple times.
                                      example: --tensor-type attn_q=q8_0
  --tensor-type-file tensor_types.txt
                                      list of tensors to quantize to a specific ggml_type
                                      this is an advanced option to selectively quantize a long list of tensors.
                                      the file should use the same format as above, separated by spaces or newlines.
  --prune-layers L0,L1,L2...
                                      comma-separated list of layer numbers to prune from the model
                                      WARNING: this is an advanced option, use with care.
  --keep-split
                                      generate quantized model in the same shards as input
  --override-kv KEY=TYPE:VALUE
                                      override model metadata by key in the quantized model. may be specified multiple times.
                                      WARNING: this is an advanced option, use with care.
  --dry-run
                                      calculate and show the final quantization size without performing quantization
                                      example: llama-quantize --dry-run model-f32.gguf Q4_K
  --max-buffer-size MiB
                                      max amount of tensor rows kept in memory while quantizing one tensor (default: 8192)
                                      lower it to quantize models with very large tensors on a machine with little RAM

note: --include-weights and --exclude-weights cannot be used together

-----------------------------------------------------------------------------
 allowed quantization types
-----------------------------------------------------------------------------

";

fn quoted(w: &mut dyn Write, prefix: &str, value: &[u8], suffix: &str) {
    let _ = w.write_all(prefix.as_bytes());
    let _ = w.write_all(value);
    let _ = w.write_all(suffix.as_bytes());
}

pub fn parse_ggml_type(
    arg: &[u8],
    type_names: &[Option<Vec<u8>>],
    err: &mut dyn Write,
) -> Option<i32> {
    let arg = c_str(arg);
    if let Some(i) = type_names
        .iter()
        .position(|name| name.as_deref().is_some_and(|n| striequals(n, arg)))
    {
        return Some(i as i32);
    }
    quoted(err, "\nparse_ggml_type: invalid ggml_type '", arg, "'\n\n");
    None
}

fn parse_tensor_type(
    data: &[u8],
    type_names: &[Option<Vec<u8>>],
    console: &mut Console<'_>,
    tensor_types: &mut Vec<(Vec<u8>, i32)>,
) -> bool {
    let data = c_str(data);
    let Some(sep) = data.iter().position(|&b| b == b'=') else {
        quoted(
            console.out,
            "\nparse_tensor_type: malformed tensor type '",
            data,
            "'\n\n",
        );
        return false;
    };
    if sep == 0 {
        let _ = console
            .out
            .write_all(b"\nparse_tensor_type: missing tensor name\n\n");
        return false;
    }
    if data.len() - sep == 1 {
        let _ = console
            .out
            .write_all(b"\nparse_tensor_type: missing quantization type\n\n");
        return false;
    }
    let name = data[..sep].to_ascii_lowercase();
    let qt = &data[sep + 1..];
    let Some(ty) = parse_ggml_type(qt, type_names, console.err) else {
        quoted(
            console.out,
            "\nparse_tensor_type: invalid quantization type '",
            qt,
            "'\n\n",
        );
        return false;
    };
    tensor_types.push((name, ty));
    true
}

fn parse_tensor_type_file(
    filename: &[u8],
    type_names: &[Option<Vec<u8>>],
    console: &mut Console<'_>,
    tensor_types: &mut Vec<(Vec<u8>, i32)>,
) -> bool {
    use std::os::unix::ffi::OsStrExt;
    let path = Path::new(std::ffi::OsStr::from_bytes(c_str(filename)));
    let content = if path.is_dir() {
        Vec::new()
    } else {
        match std::fs::read(path) {
            Ok(content) => content,
            Err(e) => {
                quoted(
                    console.out,
                    "\nparse_tensor_type_file: failed to open file '",
                    c_str(filename),
                    "': ",
                );
                let _ = console
                    .out
                    .write_all(&strerror(e.raw_os_error().unwrap_or(0)));
                let _ = console.out.write_all(b"\n\n");
                return false;
            }
        }
    };
    content
        .split(|&b| is_c_space(b))
        .filter(|token| !token.is_empty())
        .all(|token| parse_tensor_type(token, type_names, console, tensor_types))
}

fn parse_layer_prune(data: &[u8], out: &mut dyn Write, prune_layers: &mut Vec<i32>) -> bool {
    for block_id in c_str(data).split(|&b| b == b',') {
        let id = stoi(block_id).unwrap_or(-1);
        if id < 0 {
            quoted(
                out,
                "\nparse_layer_prune: invalid layer id '",
                block_id,
                "'\n\n",
            );
            return false;
        }
        prune_layers.push(id);
    }
    prune_layers.sort_unstable();
    prune_layers.dedup();
    true
}

pub fn parse_options(
    argv: &[Vec<u8>],
    type_names: &[Option<Vec<u8>>],
    console: &mut Console<'_>,
) -> Result<(Options, usize), Exit> {
    if argv.len() < 3 {
        return Err(Exit::Usage);
    }
    let mut o = Options::default();
    let mut i = 1;
    while i < argv.len() && argv[i].starts_with(b"--") {
        let has_value = i + 1 < argv.len();
        match argv[i].as_slice() {
            b"--leave-output-tensor" => o.leave_output_tensor = true,
            flag @ (b"--output-tensor-type" | b"--token-embedding-type") => {
                if !has_value {
                    return Err(Exit::Usage);
                }
                i += 1;
                let ty = parse_ggml_type(&argv[i], type_names, console.err).ok_or(Exit::Usage)?;
                if flag == b"--output-tensor-type" {
                    o.output_tensor_type = Some(ty);
                } else {
                    o.token_embedding_type = Some(ty);
                }
            }
            b"--tensor-type" => {
                i += 1;
                if !has_value
                    || !parse_tensor_type(&argv[i], type_names, console, &mut o.tensor_types)
                {
                    return Err(Exit::Usage);
                }
            }
            b"--tensor-type-file" => {
                i += 1;
                if !has_value
                    || !parse_tensor_type_file(&argv[i], type_names, console, &mut o.tensor_types)
                {
                    return Err(Exit::Usage);
                }
            }
            b"--prune-layers" => {
                i += 1;
                if !has_value || !parse_layer_prune(&argv[i], console.out, &mut o.prune_layers) {
                    return Err(Exit::Usage);
                }
            }
            b"--override-kv" => {
                i += 1;
                if !has_value {
                    return Err(Exit::Usage);
                }
                o.kv_overrides
                    .push(kv::parse(&argv[i], console.err).ok_or(Exit::Usage)?);
            }
            b"--dry-run" => o.dry_run = true,
            b"--allow-requantize" => o.allow_requantize = true,
            b"--pure" => o.pure = true,
            flag @ (b"--imatrix" | b"--include-weights" | b"--exclude-weights") => {
                if !has_value {
                    return Err(Exit::Usage);
                }
                i += 1;
                let value = c_str(&argv[i]).to_vec();
                match flag {
                    b"--imatrix" => o.imatrix_file = value,
                    b"--include-weights" => o.included_weights.push(value),
                    _ => o.excluded_weights.push(value),
                }
            }
            b"--keep-split" => o.keep_split = true,
            b"--max-buffer-size" => {
                if !has_value {
                    return Err(Exit::Usage);
                }
                i += 1;
                let mib = atoi(&argv[i]);
                if mib <= 0 {
                    quoted(
                        console.err,
                        "llama_quantize: invalid --max-buffer-size '",
                        &argv[i],
                        "'\n",
                    );
                    return Err(Exit::Code(1));
                }
                o.max_buf_size = Some(mib as usize * 1024 * 1024);
            }
            _ => return Err(Exit::Usage),
        }
        i += 1;
    }

    if argv.len() - i < 2 {
        let _ = console.out.write_all(&argv[0]);
        let _ = console.out.write_all(b": bad arguments\n");
        return Err(Exit::Usage);
    }
    if !o.included_weights.is_empty() && !o.excluded_weights.is_empty() {
        return Err(Exit::Usage);
    }
    Ok((o, i))
}

pub fn resolve_target(
    argv: &[Vec<u8>],
    mut i: usize,
    o: &Options,
    err: &mut dyn Write,
) -> Result<Target, Exit> {
    const SUFFIX: &[u8] = b".gguf";
    let input = argv[i].clone();
    i += 1;

    let (output, ftype) = if let Some(ftype) = try_parse_ftype(&argv[i]) {
        let mut output = Vec::new();
        if !o.dry_run {
            if let Some(pos) = input.iter().rposition(|&b| b == b'/' || b == b'\\') {
                output.extend_from_slice(&input[..=pos]);
            }
            output.extend_from_slice(b"ggml-model-");
            output.extend_from_slice(ftype.name.as_bytes());
            if !o.keep_split {
                output.extend_from_slice(SUFFIX);
            }
        }
        i += 1;
        (output, ftype)
    } else {
        let mut output = argv[i].clone();
        if o.keep_split && output.windows(SUFFIX.len()).any(|w| w == SUFFIX) {
            output.truncate(output.len() - SUFFIX.len());
        }
        i += 1;
        let Some(arg) = argv.get(i) else {
            let _ = err.write_all(b"llama_quantize: missing ftype\n");
            return Err(Exit::Code(1));
        };
        let Some(ftype) = try_parse_ftype(arg) else {
            quoted(err, "llama_quantize: invalid ftype '", arg, "'\n");
            return Err(Exit::Code(1));
        };
        i += 1;
        (output, ftype)
    };

    let nthread = match argv.get(i) {
        Some(arg) => match stoi(arg) {
            Some(n) => Some(n),
            None => {
                quoted(err, "llama_quantize: invalid nthread '", arg, "' (stoi)\n");
                return Err(Exit::Code(1));
            }
        },
        None => None,
    };

    Ok(Target {
        input,
        output,
        ftype,
        nthread,
    })
}

pub fn same_file(a: &[u8], b: &[u8]) -> bool {
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::fs::MetadataExt;
    let meta = |p: &[u8]| std::fs::metadata(std::ffi::OsStr::from_bytes(p));
    match (meta(a), meta(b)) {
        (Ok(a), Ok(b)) => a.dev() == b.dev() && a.ino() == b.ino(),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<Vec<u8>> {
        list.iter().map(|s| s.as_bytes().to_vec()).collect()
    }

    fn names() -> Vec<Option<Vec<u8>>> {
        ["f32", "f16", "q4_0"]
            .iter()
            .map(|n| Some(n.as_bytes().to_vec()))
            .collect()
    }

    #[test]
    fn options_and_target_follow_the_cpp_parser() {
        let (mut out, mut err) = (Vec::new(), Vec::new());
        let mut console = Console {
            out: &mut out,
            err: &mut err,
        };
        let argv = args(&[
            "q",
            "--tensor-type",
            "ATTN_Q=Q4_0",
            "--prune-layers",
            "3,1,3",
            "--keep-split",
            "dir/in.gguf",
            "x.gguf.bin",
            "q4_k",
            " 4z",
        ]);
        let (o, i) = parse_options(&argv, &names(), &mut console).unwrap();
        assert_eq!(o.tensor_types, vec![(b"attn_q".to_vec(), 2)]);
        assert_eq!(o.prune_layers, vec![1, 3]);
        let t = resolve_target(&argv, i, &o, console.err).unwrap();
        assert_eq!(t.output, b"x.ggu".to_vec());
        assert_eq!(t.ftype.name, "Q4_K");
        assert_eq!(t.nthread, Some(4));

        let argv = args(&["q", "dir\\in.gguf", "Q8_0"]);
        let (o, i) = parse_options(&argv, &names(), &mut console).unwrap();
        let t = resolve_target(&argv, i, &o, console.err).unwrap();
        assert_eq!(t.output, b"dir\\ggml-model-Q8_0.gguf".to_vec());

        let argv = args(&["q", "--tensor-type", "a=bogus", "in", "Q8_0"]);
        assert_eq!(
            parse_options(&argv, &names(), &mut console).unwrap_err(),
            Exit::Usage
        );
        assert_eq!(
            String::from_utf8_lossy(&out),
            "\nparse_tensor_type: invalid quantization type 'bogus'\n\n"
        );
        assert_eq!(
            String::from_utf8_lossy(&err),
            "\nparse_ggml_type: invalid ggml_type 'bogus'\n\n"
        );
    }
}
