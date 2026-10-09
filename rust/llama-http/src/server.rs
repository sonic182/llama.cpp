use std::{
    io,
    net::ToSocketAddrs,
    os::unix::fs::FileTypeExt,
    path::Path,
    sync::{Arc, Mutex, RwLock},
    time::Duration,
};

use hyper::{server::conn::http1, service::service_fn};
use hyper_util::{
    rt::{TokioIo, TokioTimer},
    server::graceful::GracefulShutdown,
};
use socket2::{Domain, Socket, Type};
use tokio::{
    io::{AsyncRead, AsyncWrite},
    net::{TcpListener, TcpStream, UnixListener, UnixStream},
    runtime::{Builder, Runtime},
    task::{JoinHandle, JoinSet},
};
use tokio_util::sync::CancellationToken;

use crate::{
    abi::{LOG_DBG, LOG_ERR},
    config::Settings,
    host::Host,
    routes::Routes,
    service::{Shared, handle},
    statics::StaticDir,
    tls::Tls,
    write_timeout::WriteTimeout,
};

const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);

pub struct Server {
    runtime: Option<Runtime>,
    shared: Arc<Shared>,
    cancel: CancellationToken,
    main: Mutex<Option<JoinHandle<()>>>,
}

enum Listener {
    Tcp(TcpListener),
    Unix(UnixListener),
}

enum Conn {
    Tcp(TcpStream),
    Unix(UnixStream),
}

impl Listener {
    async fn accept(&self) -> io::Result<Conn> {
        match self {
            Listener::Tcp(listener) => listener.accept().await.map(|(stream, _)| Conn::Tcp(stream)),
            Listener::Unix(listener) => listener
                .accept()
                .await
                .map(|(stream, _)| Conn::Unix(stream)),
        }
    }
}

fn is_unix_socket(host: &str) -> bool {
    host.ends_with(".sock")
}

fn clear_stale_socket(path: &Path) -> io::Result<()> {
    let Ok(meta) = std::fs::symlink_metadata(path) else {
        return Ok(());
    };
    if !meta.file_type().is_socket() {
        return Ok(());
    }
    match std::os::unix::net::UnixStream::connect(path) {
        Err(err) if err.kind() == io::ErrorKind::ConnectionRefused => std::fs::remove_file(path),
        _ => Ok(()),
    }
}

fn bind_tcp(host: &str, port: u16, settings: &Settings, v6only: bool) -> io::Result<TcpListener> {
    let addr = (host, port)
        .to_socket_addrs()?
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "address did not resolve"))?;
    let socket = Socket::new(Domain::for_address(addr), Type::STREAM, None)?;
    socket.set_reuse_address(true)?;
    if settings.reuse_port {
        socket.set_reuse_port(true)?;
    }
    if v6only && addr.is_ipv6() {
        socket.set_only_v6(true)?;
    }
    socket.bind(&addr.into())?;
    socket.listen(1024)?;
    socket.set_nonblocking(true)?;
    TcpListener::from_std(socket.into())
}

impl Server {
    pub fn new(settings: Settings, host: Host) -> io::Result<Server> {
        let runtime = Builder::new_multi_thread()
            .worker_threads(settings.n_workers)
            .max_blocking_threads(settings.n_blocking)
            .thread_name("llama-http")
            .enable_io()
            .enable_time()
            .build()?;
        let statics = match &settings.static_dir {
            Some(dir) => Some(StaticDir::new(dir, &settings.prefix).map_err(|err| {
                io::Error::new(
                    err.kind(),
                    format!("static assets path not found: {dir} ({err})"),
                )
            })?),
            None => None,
        };
        let tls = match (
            settings.ssl_cert_file.is_empty(),
            settings.ssl_key_file.is_empty(),
        ) {
            (true, true) => None,
            (false, false) => Some(Tls::load(&settings.ssl_cert_file, &settings.ssl_key_file)?),
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "--ssl-cert-file and --ssl-key-file must be given together",
                ));
            }
        };
        Ok(Server {
            runtime: Some(runtime),
            shared: Arc::new(Shared {
                settings,
                host,
                routes: RwLock::new(Routes::default()),
                statics,
                tls,
            }),
            cancel: CancellationToken::new(),
            main: Mutex::new(None),
        })
    }

    pub fn route(&self, method: i32, path: &str, id: usize) -> Result<(), String> {
        let full = format!("{}{path}", self.shared.settings.prefix);
        self.shared.routes.write().unwrap().add(method, &full, id)
    }

    pub fn start(&self) -> io::Result<u16> {
        let mut main = self.main.lock().unwrap();
        if main.is_some() {
            return Err(io::Error::other("server already started"));
        }
        let runtime = self.runtime.as_ref().expect("runtime lives until drop");
        let settings = &self.shared.settings;
        let n_tcp = settings.hosts.iter().filter(|h| !is_unix_socket(h)).count();

        let (listeners, port) = runtime.block_on(async {
            let mut listeners = Vec::new();
            let mut bound_port = settings.port;
            for host in &settings.hosts {
                if is_unix_socket(host) {
                    let path = Path::new(host);
                    clear_stale_socket(path)?;
                    listeners.push(Listener::Unix(UnixListener::bind(path)?));
                    continue;
                }
                let listener = bind_tcp(host, settings.port, settings, n_tcp > 1).map_err(|e| {
                    io::Error::new(e.kind(), format!("{host}:{}: {e}", settings.port))
                })?;
                if settings.port == 0 && bound_port == 0 {
                    bound_port = listener.local_addr()?.port();
                }
                listeners.push(Listener::Tcp(listener));
            }
            io::Result::Ok((listeners, bound_port))
        })?;

        let shared = self.shared.clone();
        let cancel = self.cancel.clone();
        *main = Some(runtime.spawn(run(listeners, shared, cancel)));
        Ok(port)
    }

    pub fn stop(&self) {
        self.cancel.cancel();
    }

    pub fn join(&self) {
        let handle = self.main.lock().unwrap().take();
        if let (Some(handle), Some(runtime)) = (handle, self.runtime.as_ref()) {
            let _ = runtime.block_on(handle);
        }
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.cancel.cancel();
        if let Some(runtime) = self.runtime.take() {
            runtime.shutdown_timeout(Duration::from_secs(2));
        }
    }
}

async fn run(listeners: Vec<Listener>, shared: Arc<Shared>, cancel: CancellationToken) {
    let graceful = Arc::new(GracefulShutdown::new());
    let mut loops = JoinSet::new();
    for listener in listeners {
        loops.spawn(accept_loop(
            listener,
            shared.clone(),
            graceful.clone(),
            cancel.clone(),
        ));
    }
    while loops.join_next().await.is_some() {}
    if let Some(graceful) = Arc::into_inner(graceful) {
        graceful.shutdown().await;
    }
}

async fn accept_loop(
    listener: Listener,
    shared: Arc<Shared>,
    graceful: Arc<GracefulShutdown>,
    cancel: CancellationToken,
) {
    let mut handshakes = JoinSet::new();
    loop {
        let conn = tokio::select! {
            () = cancel.cancelled() => break,
            Some(_) = handshakes.join_next(), if !handshakes.is_empty() => continue,
            accepted = listener.accept() => accepted,
        };
        match conn {
            Ok(Conn::Tcp(stream)) => {
                let _ = stream.set_nodelay(true);
                serve_stream(stream, &shared, &graceful, &mut handshakes);
            }
            Ok(Conn::Unix(stream)) => serve_stream(stream, &shared, &graceful, &mut handshakes),
            Err(err) => {
                shared.host.log(LOG_ERR, &format!("accept failed: {err}\n"));
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        }
    }
    handshakes.abort_all();
    while handshakes.join_next().await.is_some() {}
    shared.host.log(LOG_DBG, "listener stopped\n");
}

fn serve_stream<S>(
    stream: S,
    shared: &Arc<Shared>,
    graceful: &Arc<GracefulShutdown>,
    handshakes: &mut JoinSet<()>,
) where
    S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
{
    if shared.tls.is_none() {
        spawn_connection(TokioIo::new(stream), shared, graceful);
        return;
    }
    let shared = shared.clone();
    let graceful = graceful.clone();
    handshakes.spawn(async move {
        let Some(tls) = shared.tls.as_ref() else {
            return;
        };
        let limit = shared.settings.timeout_read.unwrap_or(HANDSHAKE_TIMEOUT);
        match tokio::time::timeout(limit, tls.accept(stream)).await {
            Ok(Ok(io)) => spawn_connection(io, &shared, &graceful),
            Ok(Err(err)) => shared
                .host
                .log(LOG_DBG, &format!("tls handshake failed: {err}\n")),
            Err(_) => shared.host.log(LOG_DBG, "tls handshake timed out\n"),
        }
    });
}

fn spawn_connection<I>(io: I, shared: &Arc<Shared>, graceful: &GracefulShutdown)
where
    I: hyper::rt::Read + hyper::rt::Write + Unpin + Send + 'static,
{
    let service_shared = shared.clone();
    let service = service_fn(move |request| handle(service_shared.clone(), request));
    let mut builder = http1::Builder::new();
    builder.timer(TokioTimer::new());
    builder.title_case_headers(true);
    if let Some(timeout) = shared.settings.timeout_read {
        builder.header_read_timeout(timeout);
    }
    let io = WriteTimeout::new(io, shared.settings.timeout_write);
    let connection = graceful.watch(builder.serve_connection(io, service));
    tokio::spawn(async move {
        let _ = connection.await;
    });
}

impl Server {
    pub fn log(&self, level: i32, message: &str) {
        self.shared.host.log(level, message);
    }
}
