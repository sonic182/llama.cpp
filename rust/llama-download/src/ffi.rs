#![allow(clippy::missing_safety_doc)]

use std::ffi::{CStr, CString, c_char, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::ptr;
use std::time::Duration;

use serde_json::{Value, json};

use crate::cache;
use crate::hub::{self, HfFile};
use crate::log;
use crate::remote::{self, Callback, ContentParams, Error, Progress};
use crate::repo::{self, InvalidRepo};
use crate::select::Wanted;

pub const INVALID_REPO: &str = "error: invalid HF repo format, expected <user>/<model>[:quant]\n";
pub const NO_HOME: &str = "Failed to find $HOME directory";

#[repr(C)]
pub struct LlamaDlProgress {
    pub url: *const c_char,
    pub downloaded: u64,
    pub total: u64,
    pub cached: bool,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct LlamaDlCallback {
    pub ctx: *mut c_void,
    pub on_start: unsafe extern "C" fn(*mut c_void, *const LlamaDlProgress),
    pub on_update: unsafe extern "C" fn(*mut c_void, *const LlamaDlProgress),
    pub on_done: unsafe extern "C" fn(*mut c_void, *const LlamaDlProgress, bool),
    pub is_cancelled: unsafe extern "C" fn(*mut c_void) -> bool,
}

#[derive(Clone, Copy)]
pub struct Adapter(LlamaDlCallback);

impl Adapter {
    pub unsafe fn from_raw(callback: *const LlamaDlCallback) -> Option<Self> {
        (!callback.is_null()).then(|| Adapter(unsafe { *callback }))
    }
}

unsafe impl Send for Adapter {}
unsafe impl Sync for Adapter {}

fn with_progress(p: &Progress, f: impl FnOnce(*const LlamaDlProgress)) {
    let url = CString::new(p.url.replace('\0', "")).unwrap_or_default();
    let raw = LlamaDlProgress {
        url: url.as_ptr(),
        downloaded: p.downloaded,
        total: p.total,
        cached: p.cached,
    };
    f(&raw)
}

impl Callback for Adapter {
    fn on_start(&self, p: &Progress) {
        with_progress(p, |raw| unsafe { (self.0.on_start)(self.0.ctx, raw) })
    }

    fn on_update(&self, p: &Progress) {
        with_progress(p, |raw| unsafe { (self.0.on_update)(self.0.ctx, raw) })
    }

    fn on_done(&self, p: &Progress, ok: bool) {
        with_progress(p, |raw| unsafe { (self.0.on_done)(self.0.ctx, raw, ok) })
    }

    fn is_cancelled(&self) -> bool {
        unsafe { (self.0.is_cancelled)(self.0.ctx) }
    }
}

struct Fail {
    kind: &'static str,
    message: String,
}

impl Fail {
    fn runtime(message: impl Into<String>) -> Self {
        Fail {
            kind: "runtime",
            message: message.into(),
        }
    }
}

impl From<InvalidRepo> for Fail {
    fn from(_: InvalidRepo) -> Self {
        Fail {
            kind: "invalid_argument",
            message: INVALID_REPO.to_string(),
        }
    }
}

impl From<Error> for Fail {
    fn from(error: Error) -> Self {
        Fail::runtime(error.to_string())
    }
}

fn into_c(text: String) -> *mut c_char {
    CString::new(text).unwrap_or_default().into_raw()
}

fn envelope(f: impl FnOnce() -> Result<Value, Fail> + std::panic::UnwindSafe) -> *mut c_char {
    match catch_unwind(f) {
        Ok(Ok(value)) => into_c(json!({ "ok": true, "value": value }).to_string()),
        Ok(Err(fail)) => {
            into_c(json!({ "ok": false, "kind": fail.kind, "error": fail.message }).to_string())
        }
        Err(_) => into_c(
            json!({ "ok": false, "kind": "runtime", "error": "llama-download panicked" })
                .to_string(),
        ),
    }
}

unsafe fn arg(text: *const c_char) -> String {
    if text.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(text) }
        .to_string_lossy()
        .into_owned()
}

unsafe fn headers(
    names: *const *const c_char,
    values: *const *const c_char,
    n: usize,
) -> Vec<(String, String)> {
    (0..n)
        .map(|i| unsafe { (arg(*names.add(i)), arg(*values.add(i))) })
        .collect()
}

fn root() -> Result<PathBuf, Fail> {
    cache::cache_dir().ok_or_else(|| Fail::runtime(NO_HOME))
}

fn hf_files_json(files: &[HfFile]) -> Value {
    Value::Array(files.iter().map(hub::to_json).collect())
}

fn opt_hf_json(file: &Option<HfFile>) -> Value {
    file.as_ref().map_or(Value::Null, hub::to_json)
}

#[unsafe(no_mangle)]
pub extern "C" fn llama_dl_set_log_sink(sink: log::Sink) {
    log::set_sink(sink);
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn llama_dl_free_string(text: *mut c_char) {
    if !text.is_null() {
        drop(unsafe { CString::from_raw(text) });
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn llama_dl_free_buffer(ptr: *mut u8, len: usize) {
    if !ptr.is_null() {
        drop(unsafe { Box::from_raw(ptr::slice_from_raw_parts_mut(ptr, len)) });
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn llama_dl_remote_get(
    url: *const c_char,
    header_names: *const *const c_char,
    header_values: *const *const c_char,
    n_headers: usize,
    timeout_seconds: u64,
    max_size: usize,
    out_status: *mut i64,
    out_body: *mut *mut u8,
    out_len: *mut usize,
) -> *mut c_char {
    let url = unsafe { arg(url) };
    let params = ContentParams {
        headers: unsafe { headers(header_names, header_values, n_headers) },
        timeout: (timeout_seconds > 0).then(|| Duration::from_secs(timeout_seconds)),
        max_size,
    };
    match catch_unwind(AssertUnwindSafe(|| remote::get_content(&url, &params))) {
        Ok(Ok((status, body))) => {
            let body = body.into_boxed_slice();
            unsafe {
                *out_status = i64::from(status);
                *out_len = body.len();
                *out_body = Box::into_raw(body) as *mut u8;
            }
            ptr::null_mut()
        }
        Ok(Err(error)) => into_c(error.to_string()),
        Err(_) => into_c("llama-download panicked".to_string()),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn llama_dl_split_repo_tag(spec: *const c_char) -> *mut c_char {
    let spec = unsafe { arg(spec) };
    envelope(move || {
        let (repo, tag) = repo::split_repo_tag(&spec)?;
        Ok(json!({ "repo": repo, "tag": tag }))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn llama_dl_hf_plan(
    spec: *const c_char,
    hf_file: *const c_char,
    token: *const c_char,
    flags: u32,
) -> *mut c_char {
    let spec = unsafe { arg(spec) };
    let hf_file = unsafe { arg(hf_file) };
    let token = unsafe { arg(token) };
    envelope(move || {
        let wanted = Wanted {
            mmproj: flags & 1 != 0,
            mtp: flags & 2 != 0,
            eagle3: flags & 4 != 0,
            dflash: flags & 8 != 0,
            dspark: flags & 16 != 0,
        };
        let offline = flags & 32 != 0;
        let plan = hub::plan(&root()?, &spec, &hf_file, offline, &token, wanted)?;
        Ok(json!({
            "preset": opt_hf_json(&plan.preset),
            "primary": opt_hf_json(&plan.primary),
            "model_files": hf_files_json(&plan.model_files),
            "mmproj": opt_hf_json(&plan.mmproj),
            "mtp": opt_hf_json(&plan.mtp),
            "eagle3": opt_hf_json(&plan.eagle3),
            "dflash": opt_hf_json(&plan.dflash),
            "dspark": opt_hf_json(&plan.dspark),
        }))
    })
}

unsafe fn set_error(error: *mut *mut c_char, message: String) {
    if !error.is_null() {
        unsafe { *error = into_c(message) };
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn llama_dl_resolve_path(
    spec: *const c_char,
    file: *const c_char,
    error: *mut *mut c_char,
) -> *mut c_char {
    let spec = unsafe { arg(spec) };
    let file = unsafe { arg(file) };
    let result = catch_unwind(move || -> Result<Option<PathBuf>, Fail> {
        Ok(cache::resolve_path(&root()?, &spec, &file)?)
    });
    match result {
        Ok(Ok(Some(path))) => into_c(path.to_string_lossy().into_owned()),
        Ok(Ok(None)) => ptr::null_mut(),
        Ok(Err(fail)) => {
            unsafe { set_error(error, fail.message) };
            ptr::null_mut()
        }
        Err(_) => {
            unsafe { set_error(error, "llama-download panicked".to_string()) };
            ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn llama_dl_remove(spec: *const c_char, error: *mut *mut c_char) -> i32 {
    let spec = unsafe { arg(spec) };
    let result =
        catch_unwind(move || -> Result<bool, Fail> { Ok(cache::remove(&root()?, &spec)?) });
    match result {
        Ok(Ok(removed)) => i32::from(removed),
        Ok(Err(fail)) => {
            unsafe { set_error(error, fail.message) };
            -1
        }
        Err(_) => {
            unsafe { set_error(error, "llama-download panicked".to_string()) };
            -1
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn llama_dl_hf_cache_path(error: *mut *mut c_char) -> *mut c_char {
    match root() {
        Ok(path) => into_c(path.to_string_lossy().into_owned()),
        Err(fail) => {
            unsafe { set_error(error, fail.message) };
            ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn llama_dl_list_cached_models(
    n_models: *mut usize,
    error: *mut *mut c_char,
) -> *mut *mut c_char {
    let result = catch_unwind(|| -> Result<Vec<String>, Fail> {
        Ok(cache::cached_models(&root()?)
            .into_iter()
            .map(|m| format!("{}:{}", m.repo_id, m.tag))
            .collect())
    });
    let models = match result {
        Ok(Ok(models)) => models,
        Ok(Err(fail)) => {
            unsafe { set_error(error, fail.message) };
            Vec::new()
        }
        Err(_) => {
            unsafe { set_error(error, "llama-download panicked".to_string()) };
            Vec::new()
        }
    };
    let list: Box<[*mut c_char]> = models.into_iter().map(into_c).collect();
    unsafe { *n_models = list.len() };
    Box::into_raw(list).cast()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn llama_dl_free_strings(list: *mut *mut c_char, n: usize) {
    if list.is_null() {
        return;
    }
    let list = unsafe { Box::from_raw(ptr::slice_from_raw_parts_mut(list, n)) };
    for text in list {
        unsafe { llama_dl_free_string(text) };
    }
}
