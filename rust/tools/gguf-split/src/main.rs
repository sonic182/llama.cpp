mod args;
mod io;
mod merge;
mod split;

use std::env;
use std::ffi::OsString;
use std::process::ExitCode;

use args::{DEFAULT_SPLIT_TENSORS, Operation, Parsed};

fn print_usage(executable: &str) {
    println!();
    println!("usage: {executable} [options] GGUF_IN GGUF_OUT");
    println!();
    print!("Apply a GGUF operation on IN to OUT.");
    println!();
    println!("options:");
    println!("  -h, --help              show this help message and exit");
    println!("  --version               show version and build info");
    println!("  --split                 split GGUF to multiple GGUF (enabled by default)");
    println!("  --merge                 merge multiple GGUF to a single GGUF");
    println!(
        "  --split-max-tensors     max tensors in each split (default: {DEFAULT_SPLIT_TENSORS})"
    );
    println!("  --split-max-size N(M|G) max size per split");
    println!(
        "  --no-tensor-first-split do not add tensors to the first split (disabled by default)"
    );
    println!(
        "  --dry-run               only print out a split plan and exit, without writing any new files"
    );
    println!(
        "  --delete-splits         delete the split files during merge to free up disk space WARNING: this option is unsafe and will leave you in an unrecoverable state if something fails during the merge"
    );
    println!();
}

fn main() -> ExitCode {
    let argv: Vec<OsString> = env::args_os().collect();
    let executable = argv
        .first()
        .map(|arg| arg.to_string_lossy().into_owned())
        .unwrap_or_default();

    let params = match args::parse(&argv) {
        Ok(Parsed::Run(params)) => params,
        Ok(Parsed::Help) => {
            print_usage(&executable);
            return ExitCode::SUCCESS;
        }
        Ok(Parsed::Version) => {
            eprintln!("version: {}", llama::version());
            return ExitCode::SUCCESS;
        }
        Err(message) => {
            eprintln!("{message}");
            print_usage(&executable);
            return ExitCode::FAILURE;
        }
    };

    let result = match params.operation {
        Operation::Split => split::split(&params).map(|()| false),
        Operation::Merge => merge::merge(&params),
    };
    match result {
        Ok(false) => ExitCode::SUCCESS,
        Ok(true) => ExitCode::FAILURE,
        Err(err) => {
            eprintln!("{err}");
            ExitCode::FAILURE
        }
    }
}
