#![allow(clippy::missing_safety_doc)]

mod abi;
mod codec;
mod config;
mod host;
mod routes;
mod server;
mod service;
mod statics;
mod tls;
mod write_timeout;

use std::ffi::c_int;

pub use abi::{
    Callbacks, Config, File, Kv, LOG_DBG, LOG_ERR, LOG_INF, LOG_WRN, METHOD_DELETE, METHOD_GET,
    METHOD_POST, Request, Response, Str,
};
use config::Settings;
use host::Host;
use server::Server;

#[unsafe(no_mangle)]
pub unsafe extern "C" fn llama_http_server_new(
    config: *const Config,
    callbacks: *const Callbacks,
) -> *mut Server {
    let (config, callbacks) = unsafe { (&*config, *callbacks) };
    let host = Host::new(callbacks);
    match Server::new(unsafe { Settings::from_raw(config) }, host) {
        Ok(server) => Box::into_raw(Box::new(server)),
        Err(err) => {
            host.log(
                LOG_ERR,
                &format!("failed to create the http server: {err}\n"),
            );
            std::ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn llama_http_server_route(
    server: *const Server,
    method: c_int,
    path: Str,
    id: usize,
) -> c_int {
    let server = unsafe { &*server };
    let path = String::from_utf8_lossy(unsafe { path.bytes() });
    match server.route(method, &path, id) {
        Ok(()) => 0,
        Err(err) => {
            server.log(LOG_WRN, &format!("route {path} not registered: {err}\n"));
            -1
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn llama_http_server_start(server: *const Server, port: *mut c_int) -> c_int {
    let server = unsafe { &*server };
    match server.start() {
        Ok(bound) => {
            unsafe { *port = c_int::from(bound) };
            0
        }
        Err(err) => {
            server.log(LOG_ERR, &format!("couldn't start the http server: {err}\n"));
            -1
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn llama_http_server_stop(server: *const Server) {
    unsafe { &*server }.stop();
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn llama_http_server_join(server: *const Server) {
    unsafe { &*server }.join();
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn llama_http_server_free(server: *mut Server) {
    if !server.is_null() {
        drop(unsafe { Box::from_raw(server) });
    }
}
