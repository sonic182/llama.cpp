use std::{
    collections::HashSet,
    sync::atomic::{AtomicBool, Ordering},
    time::Duration,
};

use crate::abi::{self, Str};

pub struct Cors {
    pub origins: String,
    pub methods: String,
    pub headers: String,
    pub credentials: bool,
}

pub struct Settings {
    pub hosts: Vec<String>,
    pub port: u16,
    pub prefix: String,
    pub api_keys: Vec<Vec<u8>>,
    pub public_paths: HashSet<String>,
    pub frontend_paths: HashSet<String>,
    pub cors: Cors,
    pub timeout_read: Option<Duration>,
    pub timeout_write: Option<Duration>,
    pub reuse_port: bool,
    pub n_workers: usize,
    pub n_blocking: usize,
    pub static_dir: Option<String>,
    pub ssl_cert_file: String,
    pub ssl_key_file: String,
    ready: ReadyFlag,
}

struct ReadyFlag(*const u8);

unsafe impl Send for ReadyFlag {}
unsafe impl Sync for ReadyFlag {}

impl Settings {
    pub unsafe fn from_raw(raw: &abi::Config) -> Self {
        Settings {
            hosts: unsafe { strings(raw.hosts, raw.n_hosts) },
            port: u16::try_from(raw.port).unwrap_or(0),
            prefix: unsafe { lossy(raw.api_prefix) },
            api_keys: unsafe { slice(raw.api_keys, raw.n_api_keys) }
                .iter()
                .map(|s| unsafe { s.bytes() }.to_vec())
                .collect(),
            public_paths: unsafe { strings(raw.public_paths, raw.n_public_paths) }
                .into_iter()
                .collect(),
            frontend_paths: unsafe { strings(raw.frontend_paths, raw.n_frontend_paths) }
                .into_iter()
                .collect(),
            cors: Cors {
                origins: unsafe { lossy(raw.cors_origins) },
                methods: unsafe { lossy(raw.cors_methods) },
                headers: unsafe { lossy(raw.cors_headers) },
                credentials: raw.cors_credentials != 0,
            },
            timeout_read: u64::try_from(raw.timeout_read_sec)
                .ok()
                .filter(|&s| s > 0)
                .map(Duration::from_secs),
            timeout_write: u64::try_from(raw.timeout_write_sec)
                .ok()
                .filter(|&s| s > 0)
                .map(Duration::from_secs),
            reuse_port: raw.reuse_port != 0,
            n_workers: usize::try_from(raw.n_workers).unwrap_or(0).max(1),
            n_blocking: usize::try_from(raw.n_blocking).unwrap_or(0).max(1),
            static_dir: Some(unsafe { lossy(raw.static_dir) }).filter(|dir| !dir.is_empty()),
            ssl_cert_file: unsafe { lossy(raw.ssl_cert_file) },
            ssl_key_file: unsafe { lossy(raw.ssl_key_file) },
            ready: ReadyFlag(raw.ready),
        }
    }

    pub fn is_ready(&self) -> bool {
        if self.ready.0.is_null() {
            return true;
        }
        unsafe { AtomicBool::from_ptr(self.ready.0.cast_mut().cast()) }.load(Ordering::Acquire)
    }
}

unsafe fn slice<'a, T>(ptr: *const T, len: usize) -> &'a [T] {
    if ptr.is_null() || len == 0 {
        return &[];
    }
    unsafe { std::slice::from_raw_parts(ptr, len) }
}

unsafe fn lossy(s: Str) -> String {
    String::from_utf8_lossy(unsafe { s.bytes() }).into_owned()
}

unsafe fn strings(ptr: *const Str, len: usize) -> Vec<String> {
    unsafe { slice(ptr, len) }
        .iter()
        .map(|s| unsafe { lossy(*s) })
        .collect()
}
