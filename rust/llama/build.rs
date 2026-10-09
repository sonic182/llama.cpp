fn main() {
    if let Ok(dir) = std::env::var("DEP_LLAMA_LIB_DIR") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
        println!("cargo:lib_dir={dir}");
    }
}
