mod escape;

use std::{
    ffi::OsString,
    fs,
    io::{self, BufWriter, Read, Write},
    os::unix::ffi::OsStrExt,
    path::PathBuf,
    process::ExitCode,
};

use anyhow::{Result, anyhow, bail};
use clap::Parser;
use llama::{Backend, Model};

use escape::process_escapes;

#[derive(Parser)]
#[command(name = "llama-tokenize")]
struct Cli {
    /// model path to load
    #[arg(short, long, value_name = "FNAME", env = "LLAMA_ARG_MODEL")]
    model: Option<PathBuf>,

    /// prompt to tokenize
    #[arg(short, long, value_name = "PROMPT")]
    prompt: Option<OsString>,

    /// a file containing the prompt (default: none)
    #[arg(short, long, value_name = "FNAME")]
    file: Option<PathBuf>,

    /// whether to process escapes sequences (\n, \r, \t, \', \", \\) (default: true)
    #[arg(short = 'e', long, overrides_with = "no_escape")]
    escape: bool,

    #[arg(long, overrides_with = "escape")]
    no_escape: bool,

    /// only print the token IDs, in a Python-parseable list form like [1, 2, 3]
    #[arg(long)]
    ids: bool,

    /// read the prompt from stdin (takes precedence over -f/--file and -p/--prompt)
    #[arg(long)]
    stdin: bool,

    /// do not add a BOS token to the prompt, even if the model normally uses one
    #[arg(long)]
    no_bos: bool,

    /// do not parse special tokens (chat, tool, etc)
    #[arg(long)]
    no_parse_special: bool,

    /// print the total number of tokens
    #[arg(long)]
    show_count: bool,
}

fn read_prompt(cli: &Cli, escape: bool) -> Result<Vec<u8>> {
    let file = cli
        .file
        .as_ref()
        .filter(|path| !path.as_os_str().is_empty());
    let mut prompt = if let Some(path) = file {
        fs::read(path)
            .map_err(|_| anyhow!("could not open file '{}' for reading", path.display()))?
    } else {
        cli.prompt
            .as_ref()
            .map(|p| p.as_bytes().to_vec())
            .unwrap_or_default()
    };
    if escape {
        prompt = process_escapes(&prompt);
    }
    Ok(prompt)
}

fn run(cli: Cli) -> Result<()> {
    let escape = !cli.no_escape;
    let has_file = cli
        .file
        .as_ref()
        .is_some_and(|path| !path.as_os_str().is_empty());
    let has_prompt = cli.prompt.as_ref().is_some_and(|prompt| !prompt.is_empty());
    if !cli.stdin && !has_file && !has_prompt {
        bail!("must specify one of: --stdin, --file or --prompt");
    }

    let mut prompt = if cli.stdin {
        Vec::new()
    } else {
        read_prompt(&cli, escape)?
    };

    let backend = Backend::init();
    let model_path = cli.model.clone().unwrap_or_default();
    let model = Model::load(&backend, &model_path, true)
        .map_err(|_| anyhow!("could not load model from file '{}'.", model_path.display()))?;
    let vocab = model.vocab();

    if cli.stdin {
        io::stdin()
            .lock()
            .read_to_end(&mut prompt)
            .map_err(|_| anyhow!("could not read the entire standard input."))?;
        if escape {
            prompt = process_escapes(&prompt);
        }
    }

    let add_bos = vocab.add_bos() && !cli.no_bos;
    let tokens = vocab.tokenize(&prompt, add_bos, !cli.no_parse_special)?;

    let mut out = BufWriter::new(io::stdout().lock());
    if cli.ids {
        out.write_all(b"[")?;
        for (i, token) in tokens.iter().enumerate() {
            if i > 0 {
                out.write_all(b", ")?;
            }
            write!(out, "{token}")?;
        }
        out.write_all(b"]\n")?;
    } else {
        for &token in &tokens {
            write!(out, "{token:>6} -> '")?;
            out.write_all(&vocab.token_to_piece(token, true)?)?;
            out.write_all(b"'\n")?;
        }
    }
    if cli.show_count {
        writeln!(out, "Total number of tokens: {}", tokens.len())?;
    }
    out.flush()?;
    Ok(())
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}
