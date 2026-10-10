#![allow(clippy::missing_safety_doc)]

use std::ptr;

use crate::{params_from_cbor, params_to_cbor};

fn into_raw(data: Vec<u8>, out: *mut *mut u8, out_len: *mut usize) {
    let data = data.into_boxed_slice();
    unsafe {
        *out_len = data.len();
        *out = Box::into_raw(data).cast();
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn llama_args_params_roundtrip(
    data: *const u8,
    len: usize,
    out: *mut *mut u8,
    out_len: *mut usize,
) -> bool {
    let input = unsafe { std::slice::from_raw_parts(data, len) };
    match params_from_cbor(input) {
        Ok(params) => {
            into_raw(params_to_cbor(&params), out, out_len);
            true
        }
        Err(error) => {
            into_raw(error.into_bytes(), out, out_len);
            false
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn llama_args_free_buffer(data: *mut u8, len: usize) {
    if !data.is_null() {
        drop(unsafe { Box::from_raw(ptr::slice_from_raw_parts_mut(data, len)) });
    }
}
