use std::ffi::CStr;
use std::fs::File;
use std::io::{self, Write};

use anyhow::{Context, Result, anyhow, bail};
use llama::gguf::{GgufBuilder, GgufFile, split_path};

use crate::args::{Mode, Params};
use crate::io::{copy_range, pad, path_of, shown, write_zeros};

pub const SPLIT_NO: &CStr = c"split.no";
pub const SPLIT_COUNT: &CStr = c"split.count";
pub const SPLIT_TENSORS_COUNT: &CStr = c"split.tensors.count";

struct Split {
    out: GgufBuilder,
    tensors: Vec<usize>,
}

struct Strategy<'a> {
    params: &'a Params,
    input: &'a GgufFile,
    splits: Vec<Split>,
}

fn start_split(input: &GgufFile, i_split: usize) -> Split {
    let mut out = GgufBuilder::new();
    if i_split == 0 {
        out.copy_kv(input);
    }
    out.set_u16(SPLIT_NO, i_split as u16);
    out.set_u16(SPLIT_COUNT, 0);
    out.set_i32(SPLIT_TENSORS_COUNT, input.n_tensors() as i32);
    Split {
        out,
        tensors: Vec::new(),
    }
}

fn should_split(params: &Params, n_tensors: usize, i_tensor: usize, next_size: usize) -> bool {
    match params.mode {
        Mode::Size => next_size > params.n_bytes_split,
        Mode::Tensor => {
            i_tensor > 0
                && i_tensor < n_tensors
                && (i_tensor as i32).checked_rem(params.n_split_tensors) == Some(0)
        }
    }
}

impl<'a> Strategy<'a> {
    fn plan(params: &'a Params, input: &'a GgufFile) -> Result<Self> {
        if params.mode == Mode::Tensor && params.n_split_tensors == 0 {
            bail!("error: --split-max-tensors must not be zero");
        }
        let n_tensors = input.n_tensors();
        let mut splits = Vec::new();
        let mut current = start_split(input, 0);
        let mut roll = |current: &mut Split, allow_empty: bool| -> Result<()> {
            if current.tensors.is_empty() && !allow_empty {
                bail!(
                    "error: one of splits have 0 tensors. Maybe size or tensors limit is too small"
                );
            }
            let next = start_split(input, splits.len() + 1);
            splits.push(std::mem::replace(current, next));
            Ok(())
        };

        if params.no_tensor_first_split {
            roll(&mut current, true)?;
        }

        let mut curr_size = 0;
        for i in 0..n_tensors {
            let n_bytes = pad(input.tensor_nbytes(i));
            let next_size = curr_size + n_bytes;
            if should_split(params, n_tensors, i, next_size) {
                roll(&mut current, false)?;
                curr_size = n_bytes;
            } else {
                curr_size = next_size;
            }
            current.out.add_tensor(input, i);
            current.tensors.push(i);
        }
        splits.push(current);

        let count = splits.len() as u16;
        for split in &mut splits {
            split.out.set_u16(SPLIT_COUNT, count);
        }
        Ok(Strategy {
            params,
            input,
            splits,
        })
    }

    fn print_info(&self) {
        println!("n_split: {}", self.splits.len());
        for (i, split) in self.splits.iter().enumerate() {
            let total = split.out.meta_size()
                + split
                    .tensors
                    .iter()
                    .map(|&t| self.input.tensor_nbytes(t))
                    .sum::<usize>();
            println!(
                "split {:05}: n_tensors = {}, total_size = {}M",
                i + 1,
                split.out.n_tensors(),
                total / 1000 / 1000
            );
        }
    }

    fn write(&self, input_file: &mut File) -> Result<()> {
        let n_split = self.splits.len();
        for (i, split) in self.splits.iter().enumerate() {
            let path = split_path(&self.params.output, i as i32, n_split as i32)
                .ok_or_else(|| anyhow!("error: split path is too long"))?;
            print!("Writing file {} ... ", shown(&path));
            std::io::stdout().flush()?;

            self.write_split(split, input_file, &path)
                .with_context(|| format!("gguf_split: failed to write {}", shown(&path)))?;
            println!("done");
        }
        Ok(())
    }

    fn write_split(&self, split: &Split, input_file: &mut File, path: &[u8]) -> io::Result<()> {
        let mut out = File::create(path_of(path))?;
        out.write_all(&split.out.meta_data())?;
        for &t in &split.tensors {
            let n_bytes = self.input.tensor_nbytes(t);
            let offset = self.input.data_offset() + self.input.tensor_offset(t);
            copy_range(input_file, &mut out, offset, n_bytes)?;
            write_zeros(&mut out, pad(n_bytes) - n_bytes)?;
        }
        out.sync_all()
    }
}

pub fn split(params: &Params) -> Result<()> {
    let mut input_file = File::open(path_of(&params.input)).map_err(|_| {
        anyhow!(
            "gguf_split:  failed to open input GGUF from {}",
            shown(&params.input)
        )
    })?;
    let input = GgufFile::open(path_of(&params.input)).map_err(|_| {
        anyhow!(
            "gguf_split:  failed to load input GGUF from {}",
            shown(&params.input)
        )
    })?;

    let strategy = Strategy::plan(params, &input)?;
    let n_split = strategy.splits.len();
    strategy.print_info();

    if !params.dry_run {
        strategy.write(&mut input_file)?;
    }

    eprintln!(
        "gguf_split: {} gguf split written with a total of {} tensors.",
        n_split,
        input.n_tensors()
    );
    Ok(())
}
