use std::fs::{self, File};
use std::io::{self, Seek, SeekFrom, Write};

use anyhow::{Context, Result, anyhow};
use llama::gguf::{GgufBuilder, GgufFile, split_path, split_prefix};

use crate::args::Params;
use crate::io::{copy_range, pad, path_of, shown, write_zeros};
use crate::split::SPLIT_COUNT;

pub fn merge(params: &Params) -> Result<bool> {
    eprintln!(
        "gguf_merge: {} -> {}",
        shown(&params.input),
        shown(&params.output)
    );

    if File::open(path_of(&params.output)).is_ok() {
        return Err(anyhow!(
            "gguf_merge: output file {} already exists",
            shown(&params.output)
        ));
    }

    let mut out = GgufBuilder::new();
    let mut n_split: i32 = 1;
    let mut total_tensors = 0;
    let mut split_prefix_bytes = Vec::new();
    let mut path = params.input.clone();
    let mut files: Vec<GgufFile> = Vec::new();

    let mut i_split = 0;
    while i_split < n_split {
        if i_split > 0 {
            path = split_path(&split_prefix_bytes, i_split, n_split).unwrap_or_default();
        }
        eprint!("gguf_merge: reading metadata {} ...", shown(&path));

        let mut file = GgufFile::open(path_of(&path)).map_err(|_| {
            anyhow!(
                "\ngguf_merge:  failed to load input GGUF from {}",
                shown(&params.input)
            )
        })?;

        if i_split == 0 {
            let key = file.find_key(SPLIT_COUNT).ok_or_else(|| {
                anyhow!(
                    "\ngguf_merge: input file does not contain {} metadata",
                    SPLIT_COUNT.to_string_lossy()
                )
            })?;
            n_split = i32::from(file.val_u16(key));
            if n_split < 1 {
                return Err(anyhow!(
                    "\ngguf_merge: input file does not contain a valid split count {n_split}"
                ));
            }
            split_prefix_bytes = split_prefix(&path, 0, n_split).ok_or_else(|| {
                anyhow!(
                    "\ngguf_merge: unexpected input file name: {} i_split=0 n_split={n_split}",
                    shown(&path)
                )
            })?;
            file.set_u16(SPLIT_COUNT, 0);
            out.copy_kv(&file);
        }

        for t in 0..file.n_tensors() {
            out.add_tensor(&file, t);
        }
        total_tensors += file.n_tensors();
        files.push(file);
        eprintln!("\x1b[3Ddone");
        i_split += 1;
    }

    let mut fout = if params.dry_run {
        None
    } else {
        let fout = File::create(path_of(&params.output))
            .and_then(|mut fout| write_zeros(&mut fout, out.meta_size()).map(|()| fout))
            .with_context(|| output_error(params))?;
        Some(fout)
    };

    let mut merge_error = false;
    for (i, file) in files.into_iter().enumerate() {
        let path = split_path(&split_prefix_bytes, i as i32, n_split).unwrap_or_default();
        let mut f_input = File::open(path_of(&path)).map_err(|_| {
            anyhow!(
                "gguf_merge:  failed to open input GGUF from {}",
                shown(&path)
            )
        })?;
        eprint!("gguf_merge: writing tensors {} ...", shown(&path));

        if let Some(fout) = fout.as_mut() {
            copy_tensors(&file, &mut f_input, fout).with_context(|| {
                format!(
                    "gguf_merge: failed to copy tensors from {} to {}",
                    shown(&path),
                    shown(&params.output)
                )
            })?;
        }
        drop(file);
        drop(f_input);
        eprintln!("\x1b[3Ddone");

        if fout.is_some() && params.delete_splits {
            match fs::remove_file(path_of(&path)) {
                Ok(()) => eprintln!("gguf_merge: deleted file {}", shown(&path)),
                Err(_) => {
                    merge_error = true;
                    eprintln!("error: failed to delete {}", shown(&path));
                }
            }
        }
    }

    if let Some(mut fout) = fout {
        fout.seek(SeekFrom::Start(0))
            .and_then(|_| fout.write_all(&out.meta_data()))
            .and_then(|()| fout.sync_all())
            .with_context(|| output_error(params))?;
    }

    eprintln!(
        "gguf_merge: {} merged from {} split with {} tensors.",
        shown(&params.output),
        n_split,
        total_tensors
    );
    Ok(merge_error)
}

fn output_error(params: &Params) -> String {
    format!("gguf_merge: failed to write {}", shown(&params.output))
}

fn copy_tensors(file: &GgufFile, input: &mut File, output: &mut File) -> io::Result<()> {
    for t in 0..file.n_tensors() {
        let n_bytes = file.tensor_nbytes(t);
        let offset = file.data_offset() + file.tensor_offset(t);
        copy_range(input, output, offset, n_bytes)?;
        write_zeros(output, pad(n_bytes) - n_bytes)?;
    }
    Ok(())
}
