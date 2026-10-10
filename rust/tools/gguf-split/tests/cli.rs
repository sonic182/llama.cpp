use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
};

const N_TENSORS: usize = 3;
const TENSOR_BYTES: usize = 32;

fn push_str(buf: &mut Vec<u8>, text: &str) {
    buf.extend_from_slice(&(text.len() as u64).to_le_bytes());
    buf.extend_from_slice(text.as_bytes());
}

fn synthetic_gguf() -> Vec<u8> {
    let mut buf = b"GGUF".to_vec();
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&(N_TENSORS as u64).to_le_bytes());
    buf.extend_from_slice(&1u64.to_le_bytes());
    push_str(&mut buf, "general.alignment");
    buf.extend_from_slice(&4u32.to_le_bytes());
    buf.extend_from_slice(&32u32.to_le_bytes());
    for i in 0..N_TENSORS {
        push_str(&mut buf, &format!("t{i}"));
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&((TENSOR_BYTES / 4) as u64).to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&((i * TENSOR_BYTES) as u64).to_le_bytes());
    }
    buf.resize(buf.len().next_multiple_of(32), 0);
    for i in 0..N_TENSORS {
        buf.extend((0..TENSOR_BYTES).map(|b| (i * 50 + b) as u8));
    }
    buf
}

fn run(args: &[&Path]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_llama-gguf-split"))
        .args(args)
        .output()
        .unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("gguf-split-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn split_then_merge_preserves_tensor_data() {
    let dir = scratch("roundtrip");
    let input = dir.join("in.gguf");
    let original = synthetic_gguf();
    fs::write(&input, &original).unwrap();

    let prefix = dir.join("out");
    let out = run(&[
        Path::new("--split-max-tensors"),
        Path::new("1"),
        &input,
        &prefix,
    ]);
    assert!(out.status.success());
    let first = dir.join("out-00001-of-00003.gguf");
    for i in 1..=N_TENSORS {
        assert!(dir.join(format!("out-{i:05}-of-00003.gguf")).exists());
    }

    let merged = dir.join("merged.gguf");
    let out = run(&[Path::new("--merge"), &first, &merged]);
    assert!(out.status.success());
    let merged = fs::read(&merged).unwrap();
    let data = N_TENSORS * TENSOR_BYTES;
    assert_eq!(
        merged[merged.len() - data..],
        original[original.len() - data..]
    );

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn dry_run_writes_nothing_and_refuses_both_limits() {
    let dir = scratch("dry");
    let input = dir.join("in.gguf");
    fs::write(&input, synthetic_gguf()).unwrap();
    let prefix = dir.join("out");

    let out = run(&[Path::new("--dry-run"), &input, &prefix]);
    assert!(out.status.success());
    assert!(String::from_utf8_lossy(&out.stdout).starts_with("n_split: 1\n"));
    assert_eq!(fs::read_dir(&dir).unwrap().count(), 1);

    let out = run(&[
        Path::new("--split-max-tensors"),
        Path::new("1"),
        Path::new("--split-max-size"),
        Path::new("1M"),
        &input,
        &prefix,
    ]);
    assert!(!out.status.success());

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn write_errors_name_the_file() {
    let dir = scratch("errors");
    let input = dir.join("in.gguf");
    fs::write(&input, synthetic_gguf()).unwrap();
    let missing = dir.join("missing");

    let out = run(&[&input, &missing.join("out")]);
    assert!(!out.status.success());
    let split_out = dir.join("missing/out-00001-of-00001.gguf");
    assert!(
        String::from_utf8_lossy(&out.stderr)
            .contains(&format!("failed to write {}", split_out.display()))
    );

    let first = dir.join("out-00001-of-00001.gguf");
    assert!(run(&[&input, &dir.join("out")]).status.success());
    let merged = missing.join("merged.gguf");
    let out = run(&[Path::new("--merge"), &first, &merged]);
    assert!(!out.status.success());
    assert!(
        String::from_utf8_lossy(&out.stderr)
            .contains(&format!("failed to write {}", merged.display()))
    );

    fs::remove_dir_all(dir).unwrap();
}
