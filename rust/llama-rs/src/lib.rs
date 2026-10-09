#![allow(clippy::missing_safety_doc)]

extern crate llama_download;
extern crate llama_http;
extern crate llama_schema;

use std::ffi::c_char;

const VERSION: &std::ffi::CStr = c"llama-rs 0.1.0";

#[unsafe(no_mangle)]
pub extern "C" fn llama_rs_version() -> *const c_char {
    VERSION.as_ptr()
}
