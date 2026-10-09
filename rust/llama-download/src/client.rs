use std::future::Future;
use std::io;
use std::pin::Pin;
use std::sync::{Arc, OnceLock};
use std::task::{Context, Poll};
use std::time::Duration;

use bytes::Bytes;
use http::header::{AUTHORIZATION, COOKIE, HOST, LOCATION, PROXY_AUTHORIZATION};
use http::{HeaderMap, HeaderValue, Method, Request, StatusCode, Uri};
use http_body_util::{BodyExt, Empty};
use hyper::body::Incoming;
use hyper_util::rt::TokioIo;
use rustls::pki_types::ServerName;
use rustls_platform_verifier::BuilderVerifierExt;
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio::net::TcpStream;
use tokio::time::Sleep;
use tokio_rustls::TlsConnector;

const MAX_REDIRECTS: usize = 20;

trait Stream: AsyncRead + AsyncWrite + Send + Unpin {}

impl<T: AsyncRead + AsyncWrite + Send + Unpin> Stream for T {}

struct TimeoutIo {
    inner: Box<dyn Stream>,
    timeout: Duration,
    read_deadline: Option<Pin<Box<Sleep>>>,
    write_deadline: Option<Pin<Box<Sleep>>>,
}

fn timed_out(what: &str) -> io::Error {
    io::Error::new(io::ErrorKind::TimedOut, format!("{what} timed out"))
}

fn poll_deadline(
    deadline: &mut Option<Pin<Box<Sleep>>>,
    timeout: Duration,
    cx: &mut Context<'_>,
    what: &str,
) -> Poll<io::Error> {
    let sleep = deadline.get_or_insert_with(|| Box::pin(tokio::time::sleep(timeout)));
    match sleep.as_mut().poll(cx) {
        Poll::Ready(()) => {
            *deadline = None;
            Poll::Ready(timed_out(what))
        }
        Poll::Pending => Poll::Pending,
    }
}

impl AsyncRead for TimeoutIo {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        let this = &mut *self;
        match Pin::new(&mut this.inner).poll_read(cx, buf) {
            Poll::Ready(result) => {
                this.read_deadline = None;
                Poll::Ready(result)
            }
            Poll::Pending => {
                poll_deadline(&mut this.read_deadline, this.timeout, cx, "read").map(Err)
            }
        }
    }
}

impl AsyncWrite for TimeoutIo {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        let this = &mut *self;
        match Pin::new(&mut this.inner).poll_write(cx, buf) {
            Poll::Ready(result) => {
                this.write_deadline = None;
                Poll::Ready(result)
            }
            Poll::Pending => {
                poll_deadline(&mut this.write_deadline, this.timeout, cx, "write").map(Err)
            }
        }
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.inner).poll_flush(cx)
    }

    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.inner).poll_shutdown(cx)
    }
}

fn tls_connector() -> io::Result<TlsConnector> {
    static CONFIG: OnceLock<Result<Arc<rustls::ClientConfig>, String>> = OnceLock::new();
    CONFIG
        .get_or_init(|| {
            let provider = Arc::new(rustls::crypto::ring::default_provider());
            rustls::ClientConfig::builder_with_provider(provider)
                .with_safe_default_protocol_versions()
                .and_then(|builder| builder.with_platform_verifier())
                .map(|builder| Arc::new(builder.with_no_client_auth()))
                .map_err(|error| error.to_string())
        })
        .clone()
        .map(TlsConnector::from)
        .map_err(io::Error::other)
}

#[derive(PartialEq, Eq)]
struct Origin {
    tls: bool,
    host: String,
    port: u16,
}

impl Origin {
    fn of(uri: &Uri) -> io::Result<Origin> {
        let tls = match uri.scheme_str() {
            Some("https") => true,
            Some("http") => false,
            other => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("unsupported URL scheme: {}", other.unwrap_or_default()),
                ));
            }
        };
        let host = uri
            .host()
            .filter(|host| !host.is_empty())
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "invalid URL: no host"))?;
        Ok(Origin {
            tls,
            host: host.to_string(),
            port: uri.port_u16().unwrap_or(if tls { 443 } else { 80 }),
        })
    }

    fn bare_host(&self) -> &str {
        self.host
            .strip_prefix('[')
            .and_then(|host| host.strip_suffix(']'))
            .unwrap_or(&self.host)
    }

    fn host_header(&self) -> io::Result<HeaderValue> {
        let default_port = if self.tls { 443 } else { 80 };
        let value = if self.port == default_port {
            self.host.clone()
        } else {
            format!("{}:{}", self.host, self.port)
        };
        HeaderValue::from_str(&value).map_err(io::Error::other)
    }
}

fn redirect_target(base: &Uri, location: &str) -> io::Result<Uri> {
    let scheme = base.scheme_str().unwrap_or("http");
    let authority = base.authority().map(|a| a.as_str()).unwrap_or_default();
    let absolute = if location.contains("://") {
        location.to_string()
    } else if let Some(rest) = location.strip_prefix("//") {
        format!("{scheme}://{rest}")
    } else if location.starts_with('/') {
        format!("{scheme}://{authority}{location}")
    } else {
        let path = base.path();
        let dir = &path[..path.rfind('/').map_or(0, |i| i + 1)];
        format!("{scheme}://{authority}{dir}{location}")
    };
    absolute
        .parse()
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
}

pub struct Response {
    pub status: u16,
    pub headers: HeaderMap,
    body: Incoming,
}

impl Response {
    pub async fn chunk(&mut self) -> io::Result<Option<Bytes>> {
        while let Some(frame) = self.body.frame().await {
            let frame = frame.map_err(io::Error::other)?;
            if let Ok(data) = frame.into_data()
                && !data.is_empty()
            {
                return Ok(Some(data));
            }
        }
        Ok(None)
    }
}

pub struct Client {
    headers: HeaderMap,
    connect_timeout: Duration,
    io_timeout: Duration,
}

impl Client {
    pub fn new(headers: HeaderMap, connect_timeout: Duration, io_timeout: Duration) -> Client {
        Client {
            headers,
            connect_timeout,
            io_timeout,
        }
    }

    async fn connect(&self, origin: &Origin) -> io::Result<TimeoutIo> {
        let connect = async {
            let tcp = TcpStream::connect((origin.bare_host(), origin.port)).await?;
            tcp.set_nodelay(true)?;
            if !origin.tls {
                return Ok(Box::new(tcp) as Box<dyn Stream>);
            }
            let name = ServerName::try_from(origin.bare_host().to_string())
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?;
            let tls = tls_connector()?.connect(name, tcp).await?;
            Ok::<Box<dyn Stream>, io::Error>(Box::new(tls))
        };
        let inner = tokio::time::timeout(self.connect_timeout, connect)
            .await
            .map_err(|_| timed_out("connection"))??;
        Ok(TimeoutIo {
            inner,
            timeout: self.io_timeout,
            read_deadline: None,
            write_deadline: None,
        })
    }

    pub async fn send(&self, method: Method, url: &str, extra: HeaderMap) -> io::Result<Response> {
        let mut uri: Uri = url
            .parse()
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?;
        let mut method = method;
        let mut headers = self.headers.clone();
        headers.extend(extra);
        let mut origin = Origin::of(&uri)?;

        for _ in 0..=MAX_REDIRECTS {
            let io = self.connect(&origin).await?;
            let (mut sender, conn) = hyper::client::conn::http1::handshake(TokioIo::new(io))
                .await
                .map_err(io::Error::other)?;
            tokio::spawn(async move {
                let _ = conn.await;
            });

            let path = uri.path_and_query().map_or("/", |p| p.as_str()).to_string();
            let mut request = Request::builder()
                .method(method.clone())
                .uri(path)
                .body(Empty::<Bytes>::new())
                .map_err(io::Error::other)?;
            *request.headers_mut() = headers.clone();
            request.headers_mut().insert(HOST, origin.host_header()?);

            let response = sender
                .send_request(request)
                .await
                .map_err(io::Error::other)?;
            let status = response.status();
            let location = response
                .headers()
                .get(LOCATION)
                .and_then(|value| value.to_str().ok())
                .map(str::to_string);
            let redirect = matches!(
                status,
                StatusCode::MOVED_PERMANENTLY
                    | StatusCode::FOUND
                    | StatusCode::SEE_OTHER
                    | StatusCode::TEMPORARY_REDIRECT
                    | StatusCode::PERMANENT_REDIRECT
            );
            let Some(location) = location.filter(|_| redirect) else {
                let (parts, body) = response.into_parts();
                return Ok(Response {
                    status: parts.status.as_u16(),
                    headers: parts.headers,
                    body,
                });
            };

            let next = redirect_target(&uri, &location)?;
            let next_origin = Origin::of(&next)?;
            if next_origin != origin {
                for name in [AUTHORIZATION, PROXY_AUTHORIZATION, COOKIE] {
                    headers.remove(name);
                }
                headers.remove("cookie2");
            }
            if status == StatusCode::SEE_OTHER && method != Method::HEAD {
                method = Method::GET;
            }
            uri = next;
            origin = next_origin;
        }
        Err(io::Error::other(format!(
            "too many redirects (more than {MAX_REDIRECTS})"
        )))
    }
}
