use std::ffi::{CString, c_char, c_int};
use std::sync::OnceLock;

pub type Sink = extern "C" fn(level: c_int, message: *const c_char);

#[derive(Clone, Copy)]
pub enum Level {
    Debug = 0,
    Info = 1,
    Warn = 2,
    Error = 3,
}

static SINK: OnceLock<Sink> = OnceLock::new();

pub fn set_sink(sink: Sink) {
    let _ = SINK.set(sink);
}

pub fn emit(level: Level, message: impl AsRef<str>) {
    let Some(sink) = SINK.get() else {
        return;
    };
    let Ok(message) = CString::new(message.as_ref().replace('\0', "")) else {
        return;
    };
    sink(level as c_int, message.as_ptr());
}
