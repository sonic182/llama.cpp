use std::{
    ffi::{c_char, c_int},
    panic::{AssertUnwindSafe, catch_unwind},
    ptr, slice,
};

use crate::json_schema_to_grammar;

#[repr(C)]
pub struct Buf {
    pub data: *mut c_char,
    pub len: usize,
}

impl Buf {
    fn empty() -> Buf {
        Buf {
            data: ptr::null_mut(),
            len: 0,
        }
    }

    fn from_string(s: String) -> Buf {
        let boxed = s.into_bytes().into_boxed_slice();
        let len = boxed.len();
        Buf {
            data: Box::into_raw(boxed) as *mut c_char,
            len,
        }
    }

    unsafe fn free(&mut self) {
        if !self.data.is_null() {
            drop(unsafe {
                Box::from_raw(ptr::slice_from_raw_parts_mut(
                    self.data as *mut u8,
                    self.len,
                ))
            });
        }
        *self = Buf::empty();
    }
}

#[repr(C)]
pub struct SchemaResult {
    pub grammar: Buf,
    pub error: Buf,
    pub warnings: Buf,
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn llama_schema_to_grammar(
    schema: *const c_char,
    len: usize,
    out: *mut SchemaResult,
) -> c_int {
    let out = unsafe { &mut *out };
    out.grammar = Buf::empty();
    out.error = Buf::empty();
    out.warnings = Buf::empty();

    let bytes = unsafe { slice::from_raw_parts(schema as *const u8, len) };
    let result = catch_unwind(AssertUnwindSafe(|| match std::str::from_utf8(bytes) {
        Ok(text) => json_schema_to_grammar(text),
        Err(e) => Err(format!("JSON schema conversion failed:\n{e}")),
    }));
    match result {
        Ok(Ok(output)) => {
            out.grammar = Buf::from_string(output.grammar);
            if !output.warnings.is_empty() {
                out.warnings = Buf::from_string(output.warnings.join("; "));
            }
            0
        }
        Ok(Err(message)) => {
            out.error = Buf::from_string(message);
            1
        }
        Err(_) => {
            out.error = Buf::from_string("JSON schema conversion failed:\ninternal error".into());
            1
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn llama_schema_result_free(result: *mut SchemaResult) {
    let result = unsafe { &mut *result };
    unsafe {
        result.grammar.free();
        result.error.free();
        result.warnings.free();
    }
}
