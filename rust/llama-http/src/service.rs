use std::{
    convert::Infallible,
    pin::Pin,
    sync::{
        Arc, RwLock,
        atomic::{AtomicBool, Ordering},
    },
    task::{Context, Poll},
};

use bytes::Bytes;
use http_body_util::{BodyExt, Full, LengthLimitError, Limited, combinators::BoxBody};
use hyper::{
    HeaderMap, Method, Request, Response, StatusCode,
    body::{Body, Frame, Incoming},
    header::{CONTENT_TYPE, HeaderName, HeaderValue},
};
use tokio::sync::{mpsc, oneshot};
use tokio_util::task::TaskTracker;

use crate::{
    abi::{self, Kv, LOG_DBG, LOG_ERR, LOG_WRN, Str},
    codec::{self, Pair},
    config::Settings,
    host::{Handle, Head, Host},
    routes::Routes,
    statics::{Outcome, StaticDir},
    tls::Tls,
};

pub struct Shared {
    pub settings: Settings,
    pub host: Host,
    pub routes: RwLock<Routes>,
    pub statics: Option<StaticDir>,
    pub tls: Option<Tls>,
    pub handlers: TaskTracker,
}

type Out = Response<BoxBody<Bytes, Infallible>>;

const JSON: &str = "application/json; charset=utf-8";
const MAX_BODY: usize = 100 * 1024 * 1024;
const MAX_FORM_BODY: usize = 1024 * 1024;

struct CancelOnDrop(Arc<AtomicBool>);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

struct StreamBody {
    rx: mpsc::Receiver<Bytes>,
    _cancel: CancelOnDrop,
}

impl Body for StreamBody {
    type Data = Bytes;
    type Error = Infallible;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, Infallible>>> {
        self.get_mut()
            .rx
            .poll_recv(cx)
            .map(|chunk| chunk.map(|bytes| Ok(Frame::data(bytes))))
    }
}

struct Multipart {
    fields: Vec<Pair>,
    files: Vec<UploadedFile>,
}

struct UploadedFile {
    key: Vec<u8>,
    filename: Vec<u8>,
    content_type: Vec<u8>,
    data: Bytes,
}

struct Call {
    path: String,
    query_string: String,
    body: Bytes,
    params: Vec<Pair>,
    headers: Vec<Pair>,
    multipart: Option<Multipart>,
}

fn kvs(pairs: &[Pair]) -> Vec<Kv> {
    pairs
        .iter()
        .map(|(key, value)| Kv {
            key: Str::new(key),
            value: Str::new(value),
        })
        .collect()
}

impl Call {
    fn run<R>(&self, f: impl FnOnce(&abi::Request) -> R) -> R {
        let params = kvs(&self.params);
        let headers = kvs(&self.headers);
        let (fields, files) = match &self.multipart {
            Some(multipart) => (
                kvs(&multipart.fields),
                multipart
                    .files
                    .iter()
                    .map(|file| abi::File {
                        key: Str::new(&file.key),
                        filename: Str::new(&file.filename),
                        content_type: Str::new(&file.content_type),
                        data: Str::new(&file.data),
                    })
                    .collect(),
            ),
            None => (Vec::new(), Vec::new()),
        };
        f(&abi::Request {
            path: Str::new(self.path.as_bytes()),
            query_string: Str::new(self.query_string.as_bytes()),
            body: Str::new(&self.body),
            params: params.as_ptr(),
            n_params: params.len(),
            headers: headers.as_ptr(),
            n_headers: headers.len(),
            is_multipart: i32::from(self.multipart.is_some()),
            fields: fields.as_ptr(),
            n_fields: fields.len(),
            files: files.as_ptr(),
            n_files: files.len(),
        })
    }
}

fn error_json(code: u16, message: &str, kind: &str) -> String {
    format!(r#"{{"error":{{"code":{code},"message":"{message}","type":"{kind}"}}}}"#)
}

fn header_pair(name: &[u8], value: &[u8]) -> Option<(HeaderName, HeaderValue)> {
    Some((
        HeaderName::from_bytes(name).ok()?,
        HeaderValue::from_bytes(value).ok()?,
    ))
}

fn text(
    status: u16,
    content_type: &str,
    body: impl Into<Bytes>,
    extra: &[(HeaderName, HeaderValue)],
) -> Out {
    let mut response = Response::new(Full::new(body.into()).boxed());
    *response.status_mut() =
        StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let headers = response.headers_mut();
    headers.insert("server", HeaderValue::from_static("llama.cpp"));
    for (name, value) in extra {
        headers.insert(name.clone(), value.clone());
    }
    if let Ok(value) = HeaderValue::from_str(content_type) {
        headers.insert(CONTENT_TYPE, value);
    }
    response
}

fn static_response(outcome: Outcome, cors: &[(HeaderName, HeaderValue)]) -> Out {
    match outcome {
        Outcome::Forbidden => text(403, "text/plain", Bytes::new(), cors),
        Outcome::Redirect(location) => {
            let mut response = text(301, "text/html", Bytes::new(), cors);
            if let Ok(value) = HeaderValue::from_str(&location) {
                response.headers_mut().insert("location", value);
            }
            response
        }
        Outcome::NotModified { etag } => {
            let mut response = Response::new(Full::new(Bytes::new()).boxed());
            *response.status_mut() = StatusCode::NOT_MODIFIED;
            let headers = response.headers_mut();
            headers.insert("server", HeaderValue::from_static("llama.cpp"));
            for (name, value) in cors {
                headers.insert(name.clone(), value.clone());
            }
            if let Ok(value) = HeaderValue::from_str(&etag) {
                headers.insert("etag", value);
            }
            response
        }
        Outcome::File(file) => {
            let mut response = text(200, file.content_type, file.data, cors);
            if let Ok(value) = HeaderValue::from_str(&file.etag) {
                response.headers_mut().insert("etag", value);
            }
            response
        }
    }
}

fn cors_headers(shared: &Shared, request: &HeaderMap) -> Vec<(HeaderName, HeaderValue)> {
    let cors = &shared.settings.cors;
    let origin = request
        .get("origin")
        .map(|value| value.as_bytes().to_vec())
        .unwrap_or_default();
    let allow_origin = HeaderName::from_static("access-control-allow-origin");
    let value = |bytes: &[u8]| HeaderValue::from_bytes(bytes).ok();

    let mut out = Vec::new();
    if cors.credentials && cors.origins == "*" {
        out.extend(value(&origin).map(|v| (allow_origin, v)));
    } else if cors.origins == "localhost" {
        let origin_text = String::from_utf8_lossy(&origin);
        if !origin.is_empty() && codec::origin_is_localhost(&origin_text) {
            out.extend(value(&origin).map(|v| (allow_origin, v)));
        } else if !origin.is_empty() {
            shared.host.log(
                LOG_WRN,
                &format!("(CORS) skip non-localhost origin: {origin_text}\n"),
            );
        }
    } else {
        out.extend(value(cors.origins.as_bytes()).map(|v| (allow_origin, v)));
    }
    out
}

fn is_authorized(shared: &Shared, path: &str, headers: &HeaderMap) -> bool {
    let keys = &shared.settings.api_keys;
    if keys.is_empty() || shared.settings.public_paths.contains(path) {
        return true;
    }
    let mut provided = headers
        .get("authorization")
        .map(|v| v.as_bytes())
        .filter(|v| !v.is_empty())
        .or_else(|| headers.get("x-api-key").map(|v| v.as_bytes()))
        .unwrap_or_default();
    if let Some(stripped) = provided.strip_prefix(b"Bearer ") {
        provided = stripped;
    }
    keys.iter().any(|key| key == provided)
}

pub async fn handle(shared: Arc<Shared>, request: Request<Incoming>) -> Result<Out, Infallible> {
    Ok(respond(shared, request).await)
}

async fn respond(shared: Arc<Shared>, request: Request<Incoming>) -> Out {
    let (parts, body) = request.into_parts();
    let path =
        String::from_utf8_lossy(&codec::decode(parts.uri.path().as_bytes(), false)).into_owned();
    let mut cors = cors_headers(&shared, &parts.headers);

    if parts.method == Method::OPTIONS {
        let settings = &shared.settings.cors;
        let credentials = if settings.credentials {
            "true"
        } else {
            "false"
        };
        for (name, value) in [
            ("access-control-allow-credentials", credentials),
            ("access-control-allow-methods", settings.methods.as_str()),
            ("access-control-allow-headers", settings.headers.as_str()),
        ] {
            cors.extend(header_pair(name.as_bytes(), value.as_bytes()));
        }
        return text(200, "text/html", Bytes::new(), &cors);
    }

    if !shared.settings.is_ready() && !shared.settings.frontend_paths.contains(&path) {
        let body = error_json(503, "Loading model", "unavailable_error");
        return text(503, JSON, body, &cors);
    }
    if !is_authorized(&shared, &path, &parts.headers) {
        shared.host.log(LOG_WRN, "unauthorized: Invalid API Key\n");
        let body = error_json(401, "Invalid API Key", "authentication_error");
        return text(401, JSON, body, &cors);
    }

    if let Some(statics) = &shared.statics
        && matches!(parts.method, Method::GET | Method::HEAD)
    {
        let if_none_match = parts
            .headers
            .get("if-none-match")
            .map(HeaderValue::as_bytes);
        if let Some(outcome) = statics.lookup(&path, if_none_match).await {
            return static_response(outcome, &cors);
        }
    }

    let route = shared.routes.read().unwrap().find(&parts.method, &path);
    let Some((route_id, path_params)) = route else {
        let body = error_json(404, "File Not Found", "not_found_error");
        return text(404, JSON, body, &cors);
    };

    if body.size_hint().lower() > MAX_BODY as u64 {
        return text(413, "text/plain", "Payload Too Large", &cors);
    }
    let body = match Limited::new(body, MAX_BODY).collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(err) if err.is::<LengthLimitError>() => {
            return text(413, "text/plain", "Payload Too Large", &cors);
        }
        Err(_) => return text(400, "text/plain", "Bad Request", &cors),
    };

    let content_type = parts
        .headers
        .get(CONTENT_TYPE)
        .map(|v| String::from_utf8_lossy(v.as_bytes()).into_owned())
        .unwrap_or_default();

    let mut query = parts
        .uri
        .query()
        .map(codec::parse_query)
        .unwrap_or_default();
    let mut multipart = None;
    if parts.method == Method::POST {
        if content_type.starts_with("application/x-www-form-urlencoded") {
            if body.len() > MAX_FORM_BODY {
                return text(413, "text/plain", "Payload Too Large", &cors);
            }
            query.extend(codec::parse_query(&String::from_utf8_lossy(&body)));
            query.sort_by(|a, b| a.0.cmp(&b.0));
        } else if content_type.starts_with("multipart/form-data") {
            match parse_multipart(&content_type, body.clone()).await {
                Ok(parsed) => multipart = Some(parsed),
                Err(err) => {
                    shared
                        .host
                        .log(LOG_DBG, &format!("invalid multipart body: {err}\n"));
                    return text(400, "text/plain", "Bad Request", &cors);
                }
            }
        }
    }

    let call = Call {
        path,
        query_string: codec::build_query_string(&query),
        body,
        params: query.into_iter().chain(path_params).collect(),
        headers: parts
            .headers
            .iter()
            .map(|(name, value)| (name.as_str().as_bytes().to_vec(), value.as_bytes().to_vec()))
            .collect(),
        multipart,
    };

    let cancelled = Arc::new(AtomicBool::new(false));
    let cancel = CancelOnDrop(cancelled.clone());
    let (head_tx, head_rx) = oneshot::channel();
    let (chunk_tx, chunk_rx) = mpsc::channel(1);
    let host = shared.host;
    shared.handlers.spawn_blocking(move || {
        serve_blocking(host, route_id, call, cancelled, head_tx, chunk_tx);
    });

    let Ok(Some((head, is_stream))) = head_rx.await else {
        shared.host.log(LOG_ERR, "handler failed\n");
        return text(500, "text/plain", "Internal Server Error", &cors);
    };

    let body = if is_stream {
        StreamBody {
            rx: chunk_rx,
            _cancel: cancel,
        }
        .boxed()
    } else {
        Full::new(Bytes::from(head.data)).boxed()
    };
    let mut response = Response::new(body);
    *response.status_mut() =
        StatusCode::from_u16(head.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let headers = response.headers_mut();
    headers.insert("server", HeaderValue::from_static("llama.cpp"));
    for (name, value) in cors {
        headers.insert(name, value);
    }
    for (name, value) in &head.headers {
        if let Some((name, value)) = header_pair(name, value) {
            headers.insert(name, value);
        }
    }
    if is_stream {
        headers.insert("x-accel-buffering", HeaderValue::from_static("no"));
    }
    if let Ok(value) = HeaderValue::from_bytes(&head.content_type)
        && !value.is_empty()
    {
        headers.insert(CONTENT_TYPE, value);
    }
    response
}

fn serve_blocking(
    host: Host,
    route_id: usize,
    call: Call,
    cancelled: Arc<AtomicBool>,
    head_tx: oneshot::Sender<Option<(Head, bool)>>,
    chunk_tx: mpsc::Sender<Bytes>,
) {
    let Some(mut head) = call.run(|request| host.dispatch(route_id, request, &cancelled)) else {
        let _ = head_tx.send(None);
        return;
    };
    drop(call);

    let stream = head.stream.take();
    let is_stream = stream.is_some();
    if head_tx.send(Some((head, is_stream))).is_err() {
        if let Some(handle) = stream {
            host.release(handle);
        }
        return;
    }
    if let Some(handle) = stream {
        pump(&host, handle, &cancelled, &chunk_tx);
    }
}

fn pump(host: &Host, handle: Handle, cancelled: &AtomicBool, chunk_tx: &mpsc::Sender<Bytes>) {
    loop {
        let (has_next, chunk) = host.next(&handle);
        if !chunk.is_empty() && chunk_tx.blocking_send(Bytes::from(chunk)).is_err() {
            cancelled.store(true, Ordering::Release);
            break;
        }
        if !has_next {
            break;
        }
    }
    host.release(handle);
}

async fn parse_multipart(content_type: &str, body: Bytes) -> Result<Multipart, multer::Error> {
    let boundary = multer::parse_boundary(content_type)?;
    let stream = futures_util::stream::once(async move { Ok::<_, Infallible>(body) });
    let mut parser = multer::Multipart::new(stream, boundary);
    let mut multipart = Multipart {
        fields: Vec::new(),
        files: Vec::new(),
    };
    while let Some(field) = parser.next_field().await? {
        let key = field.name().unwrap_or_default().as_bytes().to_vec();
        let filename = field
            .file_name()
            .filter(|name| !name.is_empty())
            .map(|name| name.as_bytes().to_vec());
        let content_type = field
            .content_type()
            .map(|mime| mime.to_string().into_bytes())
            .unwrap_or_default();
        let data = field.bytes().await?;
        match filename {
            Some(filename) => multipart.files.push(UploadedFile {
                key,
                filename,
                content_type,
                data,
            }),
            None => multipart.fields.push((key, data.to_vec())),
        }
    }
    Ok(multipart)
}
