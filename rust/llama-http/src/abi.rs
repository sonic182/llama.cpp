use std::ffi::{c_int, c_void};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Str {
    pub ptr: *const u8,
    pub len: usize,
}

impl Str {
    pub const EMPTY: Str = Str {
        ptr: std::ptr::null(),
        len: 0,
    };

    pub fn new(bytes: &[u8]) -> Self {
        Str {
            ptr: bytes.as_ptr(),
            len: bytes.len(),
        }
    }

    pub unsafe fn bytes<'a>(self) -> &'a [u8] {
        if self.ptr.is_null() || self.len == 0 {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Kv {
    pub key: Str,
    pub value: Str,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct File {
    pub key: Str,
    pub filename: Str,
    pub content_type: Str,
    pub data: Str,
}

#[repr(C)]
pub struct Config {
    pub hosts: *const Str,
    pub n_hosts: usize,
    pub port: c_int,
    pub api_prefix: Str,
    pub api_keys: *const Str,
    pub n_api_keys: usize,
    pub public_paths: *const Str,
    pub n_public_paths: usize,
    pub frontend_paths: *const Str,
    pub n_frontend_paths: usize,
    pub cors_origins: Str,
    pub cors_methods: Str,
    pub cors_headers: Str,
    pub cors_credentials: c_int,
    pub timeout_read_sec: c_int,
    pub timeout_write_sec: c_int,
    pub reuse_port: c_int,
    pub n_workers: c_int,
    pub n_blocking: c_int,
    pub ready: *const u8,
    pub static_dir: Str,
    pub ssl_cert_file: Str,
    pub ssl_key_file: Str,
}

#[repr(C)]
pub struct Request {
    pub path: Str,
    pub query_string: Str,
    pub body: Str,
    pub params: *const Kv,
    pub n_params: usize,
    pub headers: *const Kv,
    pub n_headers: usize,
    pub is_multipart: c_int,
    pub fields: *const Kv,
    pub n_fields: usize,
    pub files: *const File,
    pub n_files: usize,
}

#[repr(C)]
pub struct Response {
    pub status: c_int,
    pub content_type: Str,
    pub headers: *const Kv,
    pub n_headers: usize,
    pub data: Str,
    pub is_stream: c_int,
    pub handle: *mut c_void,
}

impl Response {
    pub const EMPTY: Response = Response {
        status: 0,
        content_type: Str::EMPTY,
        headers: std::ptr::null(),
        n_headers: 0,
        data: Str::EMPTY,
        is_stream: 0,
        handle: std::ptr::null_mut(),
    };
}

pub type DispatchFn = unsafe extern "C" fn(
    user: *mut c_void,
    route: usize,
    request: *const Request,
    cancelled: *const u8,
    response: *mut Response,
) -> c_int;
pub type NextFn = unsafe extern "C" fn(handle: *mut c_void, chunk: *mut Str) -> c_int;
pub type ReleaseFn = unsafe extern "C" fn(handle: *mut c_void);
pub type LogFn = unsafe extern "C" fn(user: *mut c_void, level: c_int, msg: Str);

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Callbacks {
    pub user: *mut c_void,
    pub dispatch: DispatchFn,
    pub next: NextFn,
    pub release: ReleaseFn,
    pub log: LogFn,
}

pub const METHOD_GET: c_int = 0;
pub const METHOD_POST: c_int = 1;
pub const METHOD_DELETE: c_int = 2;

pub const LOG_ERR: c_int = 0;
pub const LOG_WRN: c_int = 1;
pub const LOG_INF: c_int = 2;
pub const LOG_DBG: c_int = 3;
