use std::ffi::OsString;
use std::os::unix::ffi::{OsStrExt, OsStringExt};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Operation {
    Split,
    Merge,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Tensor,
    Size,
}

pub struct Params {
    pub operation: Operation,
    pub mode: Mode,
    pub n_bytes_split: usize,
    pub n_split_tensors: i32,
    pub input: Vec<u8>,
    pub output: Vec<u8>,
    pub no_tensor_first_split: bool,
    pub dry_run: bool,
    pub delete_splits: bool,
}

pub const DEFAULT_SPLIT_TENSORS: i32 = 128;

pub enum Parsed {
    Run(Params),
    Help,
    Version,
}

pub fn parse(args: &[OsString]) -> Result<Parsed, String> {
    let mut operation = None;
    let mut mode = None;
    let mut n_bytes_split = 0;
    let mut n_split_tensors = DEFAULT_SPLIT_TENSORS;
    let mut no_tensor_first_split = false;
    let mut dry_run = false;
    let mut delete_splits = false;
    let mut invalid_param = false;
    let mut arg = String::new();

    let mut idx = 1;
    while idx < args.len() && args[idx].as_bytes().starts_with(b"--") {
        arg = args[idx].to_string_lossy().replace('_', "-");
        match arg.as_str() {
            "--help" => return Ok(Parsed::Help),
            "--version" => return Ok(Parsed::Version),
            "--dry-run" => dry_run = true,
            "--no-tensor-first-split" => no_tensor_first_split = true,
            "--merge" | "--split" => {
                let requested = if arg == "--merge" {
                    Operation::Merge
                } else {
                    Operation::Split
                };
                if operation.is_some_and(|op| op != requested) {
                    return Err(
                        "error: either --split or --merge can be specified, but not both".into(),
                    );
                }
                operation = Some(requested);
            }
            "--split-max-tensors" => {
                idx += 1;
                let Some(value) = args.get(idx) else {
                    invalid_param = true;
                    break;
                };
                if mode.is_some_and(|m| m != Mode::Tensor) {
                    return Err("error: either --split-max-tensors or --split-max-size can be specified, but not both".into());
                }
                mode = Some(Mode::Tensor);
                n_split_tensors = atoi(value.as_bytes());
            }
            "--split-max-size" => {
                idx += 1;
                let Some(value) = args.get(idx) else {
                    invalid_param = true;
                    break;
                };
                if mode.is_some_and(|m| m != Mode::Size) {
                    return Err("error: either --split-max-tensors or --split-max-size can be specified, but not both".into());
                }
                mode = Some(Mode::Size);
                n_bytes_split = size_to_n_bytes(value.as_bytes())?;
            }
            "--delete-splits" => delete_splits = true,
            _ => return Err(format!("error: unknown argument: {arg}")),
        }
        idx += 1;
    }

    if invalid_param {
        return Err(format!("error: invalid parameter for argument: {arg}"));
    }
    if args.len() - idx != 2 {
        return Err("error: bad arguments".into());
    }

    Ok(Parsed::Run(Params {
        operation: operation.unwrap_or(Operation::Split),
        mode: mode.unwrap_or(Mode::Tensor),
        n_bytes_split,
        n_split_tensors,
        input: args[idx].clone().into_vec(),
        output: args[idx + 1].clone().into_vec(),
        no_tensor_first_split,
        dry_run,
        delete_splits,
    }))
}

fn size_to_n_bytes(text: &[u8]) -> Result<usize, String> {
    let unit: usize = match text.last() {
        Some(b'M') => 1_000_000,
        Some(b'G') => 1_000_000_000,
        other => {
            let got = other.map(|&b| (b as char).to_string()).unwrap_or_default();
            return Err(format!(
                "error: supported units are M (megabytes) or G (gigabytes), but got: {got}"
            ));
        }
    };
    match usize::try_from(atoi(text)) {
        Ok(n) if n > 0 => Ok(n * unit),
        _ => Err("error: size must be a positive value".into()),
    }
}

fn atoi(text: &[u8]) -> i32 {
    let mut rest = text;
    while let [first, tail @ ..] = rest {
        if first.is_ascii_whitespace() || *first == 0x0b {
            rest = tail;
        } else {
            break;
        }
    }
    let negative = match rest {
        [b'-', tail @ ..] => {
            rest = tail;
            true
        }
        [b'+', tail @ ..] => {
            rest = tail;
            false
        }
        _ => false,
    };
    let mut value: i64 = 0;
    for &byte in rest.iter().take_while(|b| b.is_ascii_digit()) {
        value = (value * 10 + i64::from(byte - b'0')).min(i64::from(i32::MAX) + 1);
    }
    let value = if negative { -value } else { value };
    value.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
}
