use std::ffi::OsStr;
use std::os::unix::ffi::OsStrExt;
use std::path::Path;

use llama::gguf::GgufFile;
use llama::sys;
use llama_quantize_core::imatrix::{Entry, Imatrix, parse_legacy};

const DATASETS: &std::ffi::CStr = c"imatrix.datasets";
const CHUNK_COUNT: &std::ffi::CStr = c"imatrix.chunk_count";
const CHUNK_SIZE: &std::ffi::CStr = c"imatrix.chunk_size";
const FUNC: &str = "common_imatrix_load";

fn line(prefix: &str, value: &[u8], suffix: &str) -> Vec<u8> {
    let mut out = prefix.as_bytes().to_vec();
    out.extend_from_slice(value);
    out.extend_from_slice(suffix.as_bytes());
    out
}

fn load_legacy(fname: &[u8], log: &mut dyn FnMut(&[u8])) -> Option<Imatrix> {
    let path = Path::new(OsStr::from_bytes(fname));
    let data = if path.is_dir() {
        Vec::new()
    } else {
        match std::fs::read(path) {
            Ok(data) => data,
            Err(_) => {
                log(&line(
                    "common_imatrix_load_legacy: failed to open ",
                    fname,
                    "\n",
                ));
                return None;
            }
        }
    };
    parse_legacy(&data, fname).map_err(|msg| log(&msg)).ok()
}

pub fn load(fname: &[u8], log: &mut dyn FnMut(&[u8])) -> Option<Imatrix> {
    let Ok(file) = GgufFile::open_with_data(Path::new(OsStr::from_bytes(fname))) else {
        return load_legacy(fname, log);
    };

    if file.n_tensors() < 1 {
        log(&line(&format!("{FUNC}: no data in file "), fname, "\n"));
        return None;
    }

    let mut imatrix = Imatrix::default();
    let datasets = file.find_key(DATASETS);
    let chunk_count = file.find_key(CHUNK_COUNT);
    let chunk_size = file.find_key(CHUNK_SIZE);

    if let Some(key) = datasets
        && file.kv_type(key) == sys::gguf_type::GGUF_TYPE_ARRAY
        && file.arr_type(key) == sys::gguf_type::GGUF_TYPE_STRING
    {
        for i in 0..file.arr_n(key) {
            imatrix
                .datasets
                .push(file.arr_str(key, i).to_bytes().to_vec());
        }
    }

    imatrix.has_metadata = datasets.is_some() && chunk_count.is_some() && chunk_size.is_some();
    imatrix.chunk_count = chunk_count.map_or(0, |k| file.val_u32(k) as i32);
    imatrix.chunk_size = chunk_size.map_or(0, |k| file.val_u32(k) as i32);

    let mut pairs = std::collections::BTreeMap::<Vec<u8>, (Option<_>, Option<_>)>::new();
    for tensor in file.meta_tensors() {
        let name = tensor.name().to_bytes().to_vec();
        if name.is_empty() {
            continue;
        }
        if let Some(base) = name.strip_suffix(b".in_sum2") {
            pairs.entry(base.to_vec()).or_default().0 = Some(tensor);
        } else if let Some(base) = name.strip_suffix(b".counts") {
            pairs.entry(base.to_vec()).or_default().1 = Some(tensor);
        }
    }

    for (name, pair) in pairs {
        let (Some(sums), Some(counts)) = pair else {
            log(&line(
                &format!("{FUNC}: mismatched sums and counts for "),
                &name,
                "\n",
            ));
            return None;
        };
        let (Some(sums), Some(counts)) = (sums.f32_data(), counts.f32_data()) else {
            log(&line(
                &format!("{FUNC}: sums and counts for "),
                &name,
                " must be F32\n",
            ));
            return None;
        };
        imatrix.entries.insert(
            name,
            Entry {
                sums: sums.to_vec(),
                counts: counts.iter().map(|&c| c.round() as i64).collect(),
            },
        );
    }

    Some(imatrix)
}
