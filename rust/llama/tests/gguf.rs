use std::path::PathBuf;

use llama::gguf::GgufFile;

fn put_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn put_tensor(out: &mut Vec<u8>, name: &str, ne: u64, ty: u32, offset: u64) {
    put_str(out, name);
    out.extend_from_slice(&1u32.to_le_bytes());
    out.extend_from_slice(&ne.to_le_bytes());
    out.extend_from_slice(&ty.to_le_bytes());
    out.extend_from_slice(&offset.to_le_bytes());
}

fn packed_file() -> PathBuf {
    let mut out = b"GGUF".to_vec();
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&2i64.to_le_bytes());
    out.extend_from_slice(&2i64.to_le_bytes());

    put_str(&mut out, "general.alignment");
    out.extend_from_slice(&4u32.to_le_bytes());
    out.extend_from_slice(&1u32.to_le_bytes());

    put_str(&mut out, "names");
    out.extend_from_slice(&9u32.to_le_bytes());
    out.extend_from_slice(&8u32.to_le_bytes());
    out.extend_from_slice(&2u64.to_le_bytes());
    put_str(&mut out, "x");
    put_str(&mut out, "y");

    put_tensor(&mut out, "h", 1, 1, 0);
    put_tensor(&mut out, "f", 2, 0, 2);

    out.extend_from_slice(&[0, 0]);
    out.extend_from_slice(&1.5f32.to_le_bytes());
    out.extend_from_slice(&(-2.0f32).to_le_bytes());

    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("packed.gguf");
    std::fs::write(&path, out).unwrap();
    path
}

#[test]
fn packed_f32_tensor_and_string_array_are_read_safely() {
    let file = GgufFile::open_with_data(&packed_file()).unwrap();

    let f = file
        .meta_tensors()
        .find(|t| t.name().to_bytes() == b"f")
        .unwrap();
    assert_eq!(f.f32_data().unwrap(), [1.5, -2.0]);

    let key = file.find_key(c"names").unwrap();
    assert_eq!(file.arr_n(key), 2);
    assert_eq!(file.arr_str(key, 1), c"y");
    let out_of_range = std::panic::catch_unwind(|| file.arr_str(key, 2).to_owned());
    assert!(out_of_range.is_err());
}
