use std::env;
use std::path::PathBuf;

const LIBS: &[&str] = &["llama", "ggml", "ggml-base", "llama-ext-c"];

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let root = manifest.join("../..").canonicalize().unwrap();

    println!("cargo:rerun-if-env-changed=LLAMA_LIB_DIR");
    println!("cargo:rerun-if-changed=wrapper.h");
    println!(
        "cargo:rerun-if-changed={}",
        root.join("include/llama.h").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        root.join("common/rust-shim/llama_ext_c.h").display()
    );

    let lib_dir = env::var_os("LLAMA_LIB_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("build-rust/bin"));
    if !lib_dir.is_dir() {
        panic!(
            "llama libraries not found in {}: build with `cmake -B build-rust -DLLAMA_RUST=ON && cmake --build build-rust` or set LLAMA_LIB_DIR",
            lib_dir.display()
        );
    }

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    for lib in LIBS {
        println!("cargo:rustc-link-lib=dylib={lib}");
    }
    println!("cargo:lib_dir={}", lib_dir.display());

    let bindings = bindgen::Builder::default()
        .header(manifest.join("wrapper.h").to_str().unwrap())
        .clang_args([
            "-x".to_string(),
            "c".to_string(),
            format!("-I{}", root.join("include").display()),
            format!("-I{}", root.join("ggml/include").display()),
            format!("-I{}", root.join("common/rust-shim").display()),
        ])
        .allowlist_function("(llama|ggml|gguf)_.*")
        .allowlist_type("(llama|ggml|gguf)_.*")
        .allowlist_var("(LLAMA|GGML|GGUF)_.*")
        .constified_enum_module("(llama|ggml|gguf)_.*")
        .derive_default(true)
        .layout_tests(false)
        .generate()
        .expect("bindgen failed");

    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings.write_to_file(out.join("bindings.rs")).unwrap();
}
