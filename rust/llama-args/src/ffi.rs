#![allow(clippy::missing_safety_doc)]

use std::ffi::{CString, c_char};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr;

use llama_download::ffi::{Adapter, LlamaDlCallback};
use llama_download::remote::Callback;

use crate::models::{self, ErrorKind, Failure, Handler};
use crate::{params_from_cbor, params_to_cbor};

const SPEC_TYPES_MAX: usize = 16;

pub type SpecTypesFromGguf =
    unsafe extern "C" fn(path: *const c_char, types: *mut i32, cap: usize) -> usize;

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

fn fail(failure: Failure, out: *mut *mut u8, out_len: *mut usize, kind: *mut i32) {
    into_raw(failure.message.into_bytes(), out, out_len);
    if !kind.is_null() {
        unsafe { *kind = failure.kind as i32 };
    }
}

fn panicked() -> Failure {
    Failure {
        kind: ErrorKind::Runtime,
        message: "llama-args panicked".to_string(),
    }
}

fn decode(data: *const u8, len: usize) -> Result<crate::Params, Failure> {
    let input = unsafe { std::slice::from_raw_parts(data, len) };
    params_from_cbor(input).map_err(|message| Failure {
        kind: ErrorKind::Runtime,
        message,
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn llama_args_models_handler_init(
    data: *const u8,
    len: usize,
    use_mmproj: bool,
    error: *mut *mut u8,
    error_len: *mut usize,
    error_kind: *mut i32,
) -> *mut Handler {
    let result = catch_unwind(AssertUnwindSafe(|| {
        models::init(&decode(data, len)?, use_mmproj)
    }));
    match result.unwrap_or_else(|_| Err(panicked())) {
        Ok(handler) => Box::into_raw(Box::new(handler)),
        Err(failure) => {
            fail(failure, error, error_len, error_kind);
            ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn llama_args_models_handler_is_preset_repo(handler: *const Handler) -> bool {
    unsafe { handler.as_ref() }.is_some_and(Handler::is_preset_repo)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn llama_args_models_handler_apply(
    handler: *mut Handler,
    data: *const u8,
    len: usize,
    callback: *const LlamaDlCallback,
    spec_types_from_gguf: SpecTypesFromGguf,
    out: *mut *mut u8,
    out_len: *mut usize,
    error_kind: *mut i32,
) -> bool {
    let mut fallback = Handler::default();
    let handler = unsafe { handler.as_mut() }.unwrap_or(&mut fallback);
    let adapter = unsafe { Adapter::from_raw(callback) };
    let spec_types = |path: &str| -> Vec<i32> {
        let Ok(path) = CString::new(path) else {
            return Vec::new();
        };
        let mut types = [0i32; SPEC_TYPES_MAX];
        let n = unsafe { spec_types_from_gguf(path.as_ptr(), types.as_mut_ptr(), types.len()) };
        types[..n.min(SPEC_TYPES_MAX)].to_vec()
    };
    let result = catch_unwind(AssertUnwindSafe(|| {
        let mut params = decode(data, len)?;
        let callback = adapter.as_ref().map(|a| a as &dyn Callback);
        models::apply(handler, &mut params, callback, &spec_types)?;
        Ok(params)
    }));
    match result.unwrap_or_else(|_| Err(panicked())) {
        Ok(params) => {
            into_raw(params_to_cbor(&params), out, out_len);
            true
        }
        Err(failure) => {
            fail(failure, out, out_len, error_kind);
            false
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn llama_args_models_handler_free(handler: *mut Handler) {
    if !handler.is_null() {
        drop(unsafe { Box::from_raw(handler) });
    }
}
