use std::ffi::{CStr, CString, c_char, c_int};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr;

use llama_quantize_core::imatrix::Imatrix;

use crate::{imatrix, quantize};

#[unsafe(no_mangle)]
pub unsafe extern "C" fn llama_rs_quantize(argc: c_int, argv: *const *const c_char) -> c_int {
    let args: Vec<Vec<u8>> = (0..usize::try_from(argc).unwrap_or(0))
        .map(|i| unsafe { CStr::from_ptr(*argv.add(i)) }.to_bytes().to_vec())
        .collect();
    catch_unwind(|| quantize::run(&args)).unwrap_or(1)
}

#[repr(C)]
pub struct LlamaImatrixEntry {
    pub name: *const c_char,
    pub sums: *const f32,
    pub n_sums: usize,
    pub counts: *const i64,
    pub n_counts: usize,
}

#[repr(C)]
pub struct LlamaImatrixView {
    pub entries: *const LlamaImatrixEntry,
    pub n_entries: usize,
    pub datasets: *const *const c_char,
    pub n_datasets: usize,
    pub chunk_count: i32,
    pub chunk_size: i32,
    pub is_legacy: bool,
    pub has_metadata: bool,
}

pub struct LlamaImatrix {
    imatrix: Imatrix,
    _names: Vec<CString>,
    entries: Vec<LlamaImatrixEntry>,
    _datasets: Vec<CString>,
    dataset_ptrs: Vec<*const c_char>,
}

fn cstring(bytes: &[u8]) -> CString {
    CString::new(bytes).unwrap_or_default()
}

fn into_handle(imatrix: Imatrix) -> Box<LlamaImatrix> {
    let names: Vec<CString> = imatrix.entries.keys().map(|n| cstring(n)).collect();
    let entries = imatrix
        .entries
        .values()
        .zip(&names)
        .map(|(e, name)| LlamaImatrixEntry {
            name: name.as_ptr(),
            sums: e.sums.as_ptr(),
            n_sums: e.sums.len(),
            counts: e.counts.as_ptr(),
            n_counts: e.counts.len(),
        })
        .collect();
    let datasets: Vec<CString> = imatrix.datasets.iter().map(|d| cstring(d)).collect();
    let dataset_ptrs = datasets.iter().map(|d| d.as_ptr()).collect();
    Box::new(LlamaImatrix {
        imatrix,
        _names: names,
        entries,
        _datasets: datasets,
        dataset_ptrs,
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn llama_imatrix_load(
    fname: *const c_char,
    log_err: Option<unsafe extern "C" fn(*const c_char)>,
) -> *mut LlamaImatrix {
    let fname = unsafe { CStr::from_ptr(fname) }.to_bytes();
    let mut log = |msg: &[u8]| {
        if let Some(log_err) = log_err {
            let msg = cstring(msg);
            unsafe { log_err(msg.as_ptr()) };
        }
    };
    match catch_unwind(AssertUnwindSafe(|| imatrix::load(fname, &mut log))) {
        Ok(Some(imatrix)) => Box::into_raw(into_handle(imatrix)),
        _ => ptr::null_mut(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn llama_imatrix_get(handle: *const LlamaImatrix) -> LlamaImatrixView {
    let h = unsafe { &*handle };
    LlamaImatrixView {
        entries: h.entries.as_ptr(),
        n_entries: h.entries.len(),
        datasets: h.dataset_ptrs.as_ptr(),
        n_datasets: h.dataset_ptrs.len(),
        chunk_count: h.imatrix.chunk_count,
        chunk_size: h.imatrix.chunk_size,
        is_legacy: h.imatrix.is_legacy,
        has_metadata: h.imatrix.has_metadata,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn llama_imatrix_free(handle: *mut LlamaImatrix) {
    if !handle.is_null() {
        drop(unsafe { Box::from_raw(handle) });
    }
}
