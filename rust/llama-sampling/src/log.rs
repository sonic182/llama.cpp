use std::ffi::{CStr, CString, c_char, c_int};
use std::sync::OnceLock;

pub type Sink = extern "C" fn(level: c_int, func: *const c_char, message: *const c_char);

pub const TRACE: c_int = 0;
#[cfg(feature = "llguidance")]
pub const ERROR: c_int = 3;

static SINK: OnceLock<Sink> = OnceLock::new();

pub fn set_sink(sink: Sink) {
    let _ = SINK.set(sink);
}

pub fn emit(level: c_int, func: &CStr, message: &str) {
    let Some(sink) = SINK.get() else {
        return;
    };
    let Ok(message) = CString::new(message) else {
        return;
    };
    sink(level, func.as_ptr(), message.as_ptr());
}

pub fn trace(func: &CStr, message: &str) {
    emit(TRACE, func, message);
}
