use std::{
    io::Write,
    path::PathBuf,
    process::{Command, Stdio},
};

fn vocab() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../models/ggml-vocab-llama-bpe.gguf")
}

fn tokenize(args: &[&str], stdin: Option<&[u8]>) -> String {
    let mut child = Command::new(env!("CARGO_BIN_EXE_llama-tokenize"))
        .arg("-m")
        .arg(vocab())
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(stdin.unwrap_or_default())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success());
    String::from_utf8(output.stdout).unwrap()
}

#[test]
fn prints_ids_with_bos_and_count() {
    let out = tokenize(&["-p", "Hello world", "--ids", "--show-count"], None);
    assert_eq!(out, "[128000, 9906, 1917]\nTotal number of tokens: 3\n");
}

#[test]
fn stdin_and_prompt_agree_and_escapes_apply() {
    let from_prompt = tokenize(&["-p", "a\\nb"], None);
    let from_stdin = tokenize(&["--stdin"], Some(b"a\\nb"));
    let unescaped = tokenize(&["--no-escape", "-p", "a\\nb"], None);
    assert_eq!(from_prompt, from_stdin);
    assert_ne!(from_prompt, unescaped);
    assert!(from_prompt.contains("'\n'"));
}

#[test]
fn repeated_options_keep_the_last_value_and_accept_leading_hyphens() {
    let out = tokenize(&["--ids", "-p", "x", "--ids", "-p", "-1"], None);
    assert_eq!(out, tokenize(&["--ids", "-p=-1"], None));
    assert_ne!(out, tokenize(&["--ids", "-p", "x"], None));
}
