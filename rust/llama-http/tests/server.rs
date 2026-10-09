use std::{
    collections::HashMap,
    ffi::{c_int, c_void},
    io::{Read, Write},
    net::TcpStream,
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
    time::{Duration, Instant},
};

use llama_http::{
    Callbacks, Config, Kv, METHOD_GET, METHOD_POST, Request, Response, Str, llama_http_server_free,
    llama_http_server_join, llama_http_server_new, llama_http_server_route,
    llama_http_server_start, llama_http_server_stop,
};

const ECHO: usize = 0;
const CHUNKS: usize = 1;
const WAIT_CANCEL: usize = 2;
const ENDLESS: usize = 3;
const FIREHOSE: usize = 4;

static BLOCK: [u8; 64 * 1024] = [b'x'; 64 * 1024];

#[derive(Default)]
struct State {
    saw_cancel: AtomicBool,
    released: AtomicUsize,
}

struct Exchange {
    state: *const State,
    cancelled: *const u8,
    route: usize,
    data: String,
    headers: Vec<Kv>,
    pos: usize,
    last: Vec<u8>,
}

fn text(s: Str) -> String {
    String::from_utf8_lossy(unsafe { s.bytes() }).into_owned()
}

fn pairs(ptr: *const Kv, len: usize) -> Vec<(String, String)> {
    if len == 0 {
        return Vec::new();
    }
    unsafe { std::slice::from_raw_parts(ptr, len) }
        .iter()
        .map(|kv| (text(kv.key), text(kv.value)))
        .collect()
}

unsafe extern "C" fn dispatch(
    user: *mut c_void,
    route: usize,
    request: *const Request,
    cancelled: *const u8,
    response: *mut Response,
) -> c_int {
    let state = user.cast::<State>().cast_const();
    let request = unsafe { &*request };
    let mut data = format!(
        "path={}\nquery={}\nbody={}\n",
        text(request.path),
        text(request.query_string),
        text(request.body)
    );
    for (key, value) in pairs(request.params, request.n_params) {
        data.push_str(&format!("param:{key}={value}\n"));
    }
    for (key, value) in pairs(request.headers, request.n_headers) {
        data.push_str(&format!("header:{key}={value}\n"));
    }

    let mut exchange = Box::new(Exchange {
        state,
        cancelled,
        route,
        data,
        headers: vec![Kv {
            key: Str::new(b"x-handler"),
            value: Str::new(b"yes"),
        }],
        pos: 0,
        last: Vec::new(),
    });

    if route == WAIT_CANCEL {
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline {
            if unsafe { AtomicBool::from_ptr(cancelled.cast_mut().cast()) }.load(Ordering::Acquire)
            {
                unsafe { &*state }.saw_cancel.store(true, Ordering::Release);
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    let is_stream = route == CHUNKS || route == ENDLESS || route == FIREHOSE;
    let response = unsafe { &mut *response };
    response.status = 200;
    response.content_type = Str::new(if is_stream {
        b"text/event-stream"
    } else {
        b"application/json; charset=utf-8"
    });
    response.headers = exchange.headers.as_ptr();
    response.n_headers = exchange.headers.len();
    response.data = Str::new(exchange.data.as_bytes());
    response.is_stream = i32::from(is_stream);
    response.handle = std::ptr::from_mut(&mut *exchange).cast();
    std::mem::forget(exchange);
    1
}

unsafe extern "C" fn next(handle: *mut c_void, chunk: *mut Str) -> c_int {
    let exchange = unsafe { &mut *handle.cast::<Exchange>() };
    let (bytes, has_next): (&[u8], bool) = match exchange.route {
        CHUNKS => {
            let parts: [&[u8]; 3] = [b"a", b"b", b"c"];
            let part = parts[exchange.pos];
            exchange.pos += 1;
            (part, exchange.pos < parts.len())
        }
        _ => {
            if exchange.route == ENDLESS {
                std::thread::sleep(Duration::from_millis(10));
            }
            let cancelled = unsafe { AtomicBool::from_ptr(exchange.cancelled.cast_mut().cast()) }
                .load(Ordering::Acquire);
            if cancelled {
                unsafe { &*exchange.state }
                    .saw_cancel
                    .store(true, Ordering::Release);
            }
            let part: &[u8] = if exchange.route == FIREHOSE {
                &BLOCK
            } else {
                b"x"
            };
            (part, !cancelled)
        }
    };
    exchange.last = bytes.to_vec();
    unsafe { *chunk = Str::new(&exchange.last) };
    c_int::from(has_next)
}

unsafe extern "C" fn release(handle: *mut c_void) {
    let exchange = unsafe { Box::from_raw(handle.cast::<Exchange>()) };
    unsafe { &*exchange.state }
        .released
        .fetch_add(1, Ordering::AcqRel);
}

unsafe extern "C" fn log(_user: *mut c_void, _level: c_int, _msg: Str) {}

struct TestServer {
    ptr: *mut c_void,
    port: u16,
    state: Box<State>,
    ready: Box<AtomicBool>,
}

fn strs(items: &[&str]) -> Vec<Str> {
    items.iter().map(|s| Str::new(s.as_bytes())).collect()
}

fn start(api_keys: &[&str], ready: bool) -> TestServer {
    start_with(api_keys, ready, "")
}

fn start_with(api_keys: &[&str], ready: bool, static_dir: &str) -> TestServer {
    launch(api_keys, ready, static_dir, ("", "")).expect("server starts")
}

fn launch(
    api_keys: &[&str],
    ready: bool,
    static_dir: &str,
    ssl: (&str, &str),
) -> Option<TestServer> {
    launch_with(
        api_keys,
        ready,
        Options {
            static_dir,
            ssl,
            ..Options::default()
        },
    )
}

#[derive(Default)]
struct Options<'a> {
    static_dir: &'a str,
    ssl: (&'a str, &'a str),
    write_timeout_sec: i32,
    host: Option<&'a str>,
}

fn launch_with(api_keys: &[&str], ready: bool, options: Options) -> Option<TestServer> {
    let Options {
        static_dir,
        ssl: (ssl_cert, ssl_key),
        write_timeout_sec,
        host,
    } = options;
    let state = Box::<State>::default();
    let ready = Box::new(AtomicBool::new(ready));
    let hosts = strs(&[host.unwrap_or("127.0.0.1")]);
    let keys = strs(api_keys);
    let public = strs(&["/health"]);
    let config = Config {
        hosts: hosts.as_ptr(),
        n_hosts: hosts.len(),
        port: 0,
        api_prefix: Str::new(b""),
        api_keys: keys.as_ptr(),
        n_api_keys: keys.len(),
        public_paths: public.as_ptr(),
        n_public_paths: public.len(),
        frontend_paths: std::ptr::null(),
        n_frontend_paths: 0,
        cors_origins: Str::new(b"*"),
        cors_methods: Str::new(b"GET, POST"),
        cors_headers: Str::new(b"*"),
        cors_credentials: 0,
        timeout_read_sec: 30,
        timeout_write_sec: write_timeout_sec,
        reuse_port: 0,
        n_workers: 2,
        n_blocking: 16,
        ready: std::ptr::from_ref(&*ready).cast(),
        static_dir: Str::new(static_dir.as_bytes()),
        ssl_cert_file: Str::new(ssl_cert.as_bytes()),
        ssl_key_file: Str::new(ssl_key.as_bytes()),
    };
    let callbacks = Callbacks {
        user: std::ptr::from_ref(&*state).cast_mut().cast(),
        dispatch,
        next,
        release,
        log,
    };
    let server = unsafe { llama_http_server_new(&config, &callbacks) };
    if server.is_null() {
        return None;
    }
    let routes = [
        (METHOD_GET, "/echo", ECHO),
        (METHOD_GET, "/health", ECHO),
        (METHOD_POST, "/slots/:id", ECHO),
        (METHOD_GET, "/chunks", CHUNKS),
        (METHOD_GET, "/wait", WAIT_CANCEL),
        (METHOD_GET, "/endless", ENDLESS),
        (METHOD_GET, "/firehose", FIREHOSE),
    ];
    for (method, path, id) in routes {
        assert_eq!(
            unsafe { llama_http_server_route(server, method, Str::new(path.as_bytes()), id) },
            0
        );
    }
    let mut port = 0;
    if unsafe { llama_http_server_start(server, &mut port) } != 0 {
        unsafe { llama_http_server_free(server) };
        return None;
    }
    Some(TestServer {
        ptr: server.cast(),
        port: u16::try_from(port).unwrap(),
        state,
        ready,
    })
}

impl Drop for TestServer {
    fn drop(&mut self) {
        let server = self.ptr.cast();
        unsafe {
            llama_http_server_stop(server);
            llama_http_server_join(server);
            llama_http_server_free(server);
        }
    }
}

struct Reply {
    head: String,
    status: u16,
    headers: HashMap<String, String>,
    body: String,
}

fn decode_chunked(mut raw: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    while let Some(line_end) = raw.windows(2).position(|w| w == b"\r\n") {
        let size =
            usize::from_str_radix(std::str::from_utf8(&raw[..line_end]).unwrap(), 16).unwrap();
        if size == 0 {
            break;
        }
        out.extend_from_slice(&raw[line_end + 2..line_end + 2 + size]);
        raw = &raw[line_end + 2 + size + 2..];
    }
    out
}

fn send(port: u16, raw: &str) -> Reply {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(10)))
        .unwrap();
    stream.write_all(raw.as_bytes()).unwrap();
    let mut buf = Vec::new();
    stream.read_to_end(&mut buf).unwrap();
    parse_reply(&buf)
}

fn parse_reply(buf: &[u8]) -> Reply {
    let split = buf.windows(4).position(|w| w == b"\r\n\r\n").unwrap();
    let head = String::from_utf8_lossy(&buf[..split]).into_owned();
    let mut lines = head.lines();
    let status = lines
        .next()
        .unwrap()
        .split(' ')
        .nth(1)
        .unwrap()
        .parse()
        .unwrap();
    let headers: HashMap<_, _> = lines
        .filter_map(|l| l.split_once(": "))
        .map(|(k, v)| (k.to_ascii_lowercase(), v.to_owned()))
        .collect();
    let raw_body = &buf[split + 4..];
    let body = if headers
        .get("transfer-encoding")
        .is_some_and(|v| v == "chunked")
    {
        decode_chunked(raw_body)
    } else {
        raw_body.to_vec()
    };
    Reply {
        head,
        status,
        headers,
        body: String::from_utf8_lossy(&body).into_owned(),
    }
}

fn get(port: u16, path: &str, extra: &str) -> Reply {
    send(
        port,
        &format!("GET {path} HTTP/1.1\r\nHost: t\r\nConnection: close\r\n{extra}\r\n"),
    )
}

fn wait_for(what: &str, condition: impl Fn() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while !condition() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(10));
    }
}

#[test]
fn passes_params_query_headers_and_body_to_the_handler() {
    let server = start(&[], true);

    let reply = get(server.port, "/echo?b=2&a=1&a=1&sp=x%20y", "X-Test: hi\r\n");
    assert_eq!(reply.status, 200);
    assert_eq!(reply.headers["server"], "llama.cpp");
    assert_eq!(reply.headers["x-handler"], "yes");
    assert_eq!(
        reply.headers["content-type"],
        "application/json; charset=utf-8"
    );
    assert!(reply.body.contains("path=/echo\n"));
    assert!(reply.body.contains("query=a=1&b=2&sp=x+y\n"));
    assert!(reply.body.contains("header:x-test=hi\n"));

    let body = r#"{"k":1}"#;
    let reply = send(
        server.port,
        &format!(
            "POST /slots/c%3Ax HTTP/1.1\r\nHost: t\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        ),
    );
    assert_eq!(reply.status, 200);
    assert!(reply.body.contains("param:id=c:x\n"));
    assert!(reply.body.contains(&format!("body={body}\n")));
}

#[test]
fn middleware_orders_preflight_state_auth_and_not_found() {
    let server = start(&["secret"], false);

    let reply = send(
        server.port,
        "OPTIONS /echo HTTP/1.1\r\nHost: t\r\nConnection: close\r\n\r\n",
    );
    assert_eq!(reply.status, 200);
    assert_eq!(reply.headers["access-control-allow-origin"], "*");
    assert!(reply.head.contains("\r\nAccess-Control-Allow-Origin: *"));
    assert_eq!(reply.headers["access-control-allow-methods"], "GET, POST");
    assert_eq!(reply.headers["access-control-allow-credentials"], "false");

    let reply = get(server.port, "/echo", "");
    assert_eq!(reply.status, 503);
    assert_eq!(
        reply.body,
        r#"{"error":{"code":503,"message":"Loading model","type":"unavailable_error"}}"#
    );

    server.ready.store(true, Ordering::Release);
    let reply = get(server.port, "/echo", "");
    assert_eq!(reply.status, 401);
    assert_eq!(
        reply.body,
        r#"{"error":{"code":401,"message":"Invalid API Key","type":"authentication_error"}}"#
    );
    assert_eq!(get(server.port, "/health", "").status, 200);
    assert_eq!(
        get(server.port, "/echo", "Authorization: Bearer secret\r\n").status,
        200
    );
    assert_eq!(
        get(server.port, "/echo", "X-Api-Key: secret\r\n").status,
        200
    );
    assert_eq!(
        get(server.port, "/echo", "Authorization: Bearer nope\r\n").status,
        401
    );

    let reply = get(server.port, "/missing", "Authorization: Bearer secret\r\n");
    assert_eq!(reply.status, 404);
    assert_eq!(
        reply.body,
        r#"{"error":{"code":404,"message":"File Not Found","type":"not_found_error"}}"#
    );
}

#[test]
fn streams_chunks_in_order_and_releases_the_exchange() {
    let server = start(&[], true);
    let reply = get(server.port, "/chunks", "");
    assert_eq!(reply.status, 200);
    assert_eq!(reply.body, "abc");
    assert_eq!(reply.headers["x-accel-buffering"], "no");
    assert_eq!(reply.headers["content-type"], "text/event-stream");
    wait_for("release", || {
        server.state.released.load(Ordering::Acquire) == 1
    });
}

#[test]
fn client_disconnect_cancels_a_pending_buffered_handler() {
    let server = start(&[], true);
    let mut stream = TcpStream::connect(("127.0.0.1", server.port)).unwrap();
    stream
        .write_all(b"GET /wait HTTP/1.1\r\nHost: t\r\n\r\n")
        .unwrap();
    std::thread::sleep(Duration::from_millis(200));
    drop(stream);
    wait_for("cancellation", || {
        server.state.saw_cancel.load(Ordering::Acquire)
    });
    wait_for("release", || {
        server.state.released.load(Ordering::Acquire) == 1
    });
}

#[test]
fn client_disconnect_stops_an_endless_stream() {
    let server = start(&[], true);
    let mut stream = TcpStream::connect(("127.0.0.1", server.port)).unwrap();
    stream
        .write_all(b"GET /endless HTTP/1.1\r\nHost: t\r\n\r\n")
        .unwrap();
    let mut buf = [0u8; 256];
    assert!(stream.read(&mut buf).unwrap() > 0);
    drop(stream);
    wait_for("cancellation", || {
        server.state.saw_cancel.load(Ordering::Acquire)
    });
    wait_for("release", || {
        server.state.released.load(Ordering::Acquire) == 1
    });
}

#[test]
fn stalled_reader_hits_the_write_timeout_and_cancels_the_stream() {
    let server = launch_with(
        &[],
        true,
        Options {
            write_timeout_sec: 1,
            ..Options::default()
        },
    )
    .expect("server starts");
    let mut stream = TcpStream::connect(("127.0.0.1", server.port)).unwrap();
    stream
        .write_all(b"GET /firehose HTTP/1.1\r\nHost: t\r\n\r\n")
        .unwrap();
    wait_for("release after the write timeout", || {
        server.state.released.load(Ordering::Acquire) == 1
    });
}

#[test]
fn stale_unix_socket_is_replaced_but_a_live_one_is_kept() {
    let path = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("llama-http-stale.sock");
    let host = path.to_str().unwrap();
    let _ = std::fs::remove_file(&path);
    drop(std::os::unix::net::UnixListener::bind(&path).unwrap());
    assert!(path.exists());

    let options = || Options {
        host: Some(host),
        ..Options::default()
    };
    let server = launch_with(&[], true, options()).expect("stale socket is replaced");
    let mut stream = std::os::unix::net::UnixStream::connect(&path).unwrap();
    stream
        .write_all(b"GET /echo HTTP/1.1\r\nHost: t\r\nConnection: close\r\n\r\n")
        .unwrap();
    let mut buf = Vec::new();
    stream.read_to_end(&mut buf).unwrap();
    assert_eq!(parse_reply(&buf).status, 200);

    assert!(launch_with(&[], true, options()).is_none());
    drop(server);
}

#[test]
fn serves_static_files_before_routes() {
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("static-assets");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("sub")).unwrap();
    std::fs::write(dir.join("index.html"), "<h1>hi</h1>").unwrap();
    std::fs::write(dir.join("app.js"), "let x = 1;").unwrap();
    std::fs::write(dir.join("sub").join("index.html"), "sub").unwrap();

    let server = start_with(&[], true, dir.to_str().unwrap());

    let reply = get(server.port, "/", "");
    assert_eq!(reply.status, 200);
    assert_eq!(reply.headers["content-type"], "text/html");
    assert_eq!(reply.body, "<h1>hi</h1>");

    let reply = get(server.port, "/app.js", "");
    assert_eq!(reply.headers["content-type"], "text/javascript");
    let etag = reply.headers["etag"].clone();
    let reply = get(
        server.port,
        "/app.js",
        &format!("If-None-Match: {etag}\r\n"),
    );
    assert_eq!(reply.status, 304);
    assert!(reply.body.is_empty());

    assert_eq!(get(server.port, "/sub", "").status, 301);
    assert_eq!(get(server.port, "/sub/", "").body, "sub");
    assert_eq!(get(server.port, "/missing.css", "").status, 404);
    assert_eq!(get(server.port, "/../Cargo.toml", "").status, 404);
    assert!(get(server.port, "/echo", "").body.contains("path=/echo\n"));
}

#[cfg(feature = "tls")]
mod tls {
    use std::{path::PathBuf, sync::Arc};

    use rustls::{
        ClientConfig, ClientConnection, RootCertStore, StreamOwned, crypto::ring,
        pki_types::ServerName,
    };

    use super::*;

    struct Identity {
        dir: PathBuf,
        cert: String,
        key: String,
        client: Arc<ClientConfig>,
    }

    fn identity(name: &str) -> Identity {
        let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let certified = rcgen::generate_simple_self_signed(vec!["localhost".to_owned()]).unwrap();
        let cert = dir.join("cert.pem");
        let key = dir.join("key.pem");
        std::fs::write(&cert, certified.cert.pem()).unwrap();
        std::fs::write(&key, certified.signing_key.serialize_pem()).unwrap();

        let mut roots = RootCertStore::empty();
        roots.add(certified.cert.der().clone()).unwrap();
        let client = ClientConfig::builder_with_provider(Arc::new(ring::default_provider()))
            .with_safe_default_protocol_versions()
            .unwrap()
            .with_root_certificates(roots)
            .with_no_client_auth();

        Identity {
            cert: cert.to_string_lossy().into_owned(),
            key: key.to_string_lossy().into_owned(),
            dir,
            client: Arc::new(client),
        }
    }

    fn https_get(port: u16, client: &Arc<ClientConfig>, path: &str) -> Reply {
        let tcp = TcpStream::connect(("127.0.0.1", port)).unwrap();
        tcp.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
        let connection =
            ClientConnection::new(client.clone(), ServerName::try_from("localhost").unwrap())
                .unwrap();
        let mut tls = StreamOwned::new(connection, tcp);
        write!(
            tls,
            "GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"
        )
        .unwrap();
        let mut buf = Vec::new();
        let _ = tls.read_to_end(&mut buf);
        parse_reply(&buf)
    }

    #[test]
    fn serves_requests_and_streams_over_tls() {
        let id = identity("tls-serve");
        let server = launch(&[], true, "", (&id.cert, &id.key)).expect("server starts");

        let reply = https_get(server.port, &id.client, "/echo?a=1");
        assert_eq!(reply.status, 200);
        assert_eq!(reply.headers["server"], "llama.cpp");
        assert!(reply.body.contains("path=/echo\n"));
        assert!(reply.body.contains("query=a=1\n"));

        let reply = https_get(server.port, &id.client, "/chunks");
        assert_eq!(reply.status, 200);
        assert_eq!(reply.body, "abc");
    }

    #[test]
    fn plain_http_on_a_tls_port_does_not_stop_the_server() {
        let id = identity("tls-plain");
        let server = launch(&[], true, "", (&id.cert, &id.key)).expect("server starts");

        let mut plain = TcpStream::connect(("127.0.0.1", server.port)).unwrap();
        plain
            .set_read_timeout(Some(Duration::from_secs(10)))
            .unwrap();
        plain
            .write_all(b"GET /echo HTTP/1.1\r\nHost: t\r\nConnection: close\r\n\r\n")
            .unwrap();
        let mut buf = Vec::new();
        let _ = plain.read_to_end(&mut buf);
        assert!(!String::from_utf8_lossy(&buf).contains("200 OK"));

        assert_eq!(https_get(server.port, &id.client, "/echo").status, 200);
    }

    #[test]
    fn rejects_unusable_certificate_configuration() {
        let id = identity("tls-invalid");
        let garbage = id.dir.join("garbage.pem");
        std::fs::write(&garbage, "not a pem file").unwrap();
        let garbage = garbage.to_string_lossy().into_owned();
        let missing = id.dir.join("missing.pem").to_string_lossy().into_owned();

        assert!(launch(&[], true, "", (&id.cert, "")).is_none());
        assert!(launch(&[], true, "", ("", &id.key)).is_none());
        assert!(launch(&[], true, "", (&missing, &id.key)).is_none());
        assert!(launch(&[], true, "", (&id.cert, &missing)).is_none());
        assert!(launch(&[], true, "", (&garbage, &id.key)).is_none());
        assert!(launch(&[], true, "", (&id.cert, &garbage)).is_none());
        assert!(launch(&[], true, "", (&id.cert, &id.key)).is_some());
    }
}

#[cfg(not(feature = "tls"))]
#[test]
fn ssl_is_refused_when_built_without_tls() {
    assert!(
        launch(
            &[],
            true,
            "",
            ("/nonexistent/cert.pem", "/nonexistent/key.pem")
        )
        .is_none()
    );
}
