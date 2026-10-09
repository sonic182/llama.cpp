use std::{
    ffi::{c_int, c_void},
    sync::atomic::AtomicBool,
};

use crate::abi::{Callbacks, Kv, Request, Response, Str};

#[derive(Clone, Copy)]
pub struct Host(Callbacks);

unsafe impl Send for Host {}
unsafe impl Sync for Host {}

pub struct Handle(*mut c_void);

unsafe impl Send for Handle {}

pub struct Head {
    pub status: u16,
    pub content_type: Vec<u8>,
    pub headers: Vec<(Vec<u8>, Vec<u8>)>,
    pub data: Vec<u8>,
    pub stream: Option<Handle>,
}

fn owned(s: Str) -> Vec<u8> {
    unsafe { s.bytes() }.to_vec()
}

impl Host {
    pub fn new(callbacks: Callbacks) -> Self {
        Host(callbacks)
    }

    pub fn log(&self, level: c_int, msg: &str) {
        unsafe { (self.0.log)(self.0.user, level, Str::new(msg.as_bytes())) }
    }

    pub fn dispatch(
        &self,
        route: usize,
        request: &Request,
        cancelled: &AtomicBool,
    ) -> Option<Head> {
        let mut response = Response::EMPTY;
        let cancelled = std::ptr::from_ref(cancelled).cast::<u8>();
        let ok =
            unsafe { (self.0.dispatch)(self.0.user, route, request, cancelled, &mut response) };
        if ok == 0 {
            return None;
        }

        let is_stream = response.is_stream != 0;
        let headers: &[Kv] = if response.headers.is_null() {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(response.headers, response.n_headers) }
        };
        let head = Head {
            status: u16::try_from(response.status).unwrap_or(500),
            content_type: owned(response.content_type),
            headers: headers
                .iter()
                .map(|kv| (owned(kv.key), owned(kv.value)))
                .collect(),
            data: if is_stream {
                Vec::new()
            } else {
                owned(response.data)
            },
            stream: is_stream.then_some(Handle(response.handle)),
        };
        if !is_stream {
            unsafe { (self.0.release)(response.handle) };
        }
        Some(head)
    }

    pub fn next(&self, handle: &Handle) -> (bool, Vec<u8>) {
        let mut chunk = Str::EMPTY;
        let has_next = unsafe { (self.0.next)(handle.0, &mut chunk) } != 0;
        (has_next, owned(chunk))
    }

    pub fn release(&self, handle: Handle) {
        unsafe { (self.0.release)(handle.0) }
    }
}
