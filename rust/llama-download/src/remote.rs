use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;

use reqwest::header::{
    AUTHORIZATION, CONTENT_LENGTH, HeaderMap, HeaderName, HeaderValue, RANGE, USER_AGENT,
};

use crate::log::{self, Level};

const MAX_ATTEMPTS: u32 = 3;
const RETRY_DELAY: Duration = Duration::from_secs(2);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(300);
const READ_TIMEOUT: Duration = Duration::from_secs(5);
const DEFAULT_USER_AGENT: &str = "llama-cpp";

#[derive(Debug, Default, Clone)]
pub struct Progress {
    pub url: String,
    pub downloaded: u64,
    pub total: u64,
    pub cached: bool,
}

pub trait Callback: Sync {
    fn on_start(&self, _progress: &Progress) {}
    fn on_update(&self, _progress: &Progress) {}
    fn on_done(&self, _progress: &Progress, _ok: bool) {}
    fn is_cancelled(&self) -> bool {
        false
    }
}

#[derive(Default)]
pub struct Options<'a> {
    pub headers: Vec<(String, String)>,
    pub bearer_token: Option<String>,
    pub offline: bool,
    pub callback: Option<&'a dyn Callback>,
}

#[derive(Default)]
pub struct ContentParams {
    pub headers: Vec<(String, String)>,
    pub timeout: Option<Duration>,
    pub max_size: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    Status(u16),
    Failed(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Status(status) => write!(f, "HTTP status code {status}"),
            Error::Failed(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for Error {}

fn failed(error: impl std::fmt::Display) -> Error {
    Error::Failed(error.to_string())
}

fn runtime() -> Result<tokio::runtime::Runtime, Error> {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(failed)
}

fn client(headers: &[(String, String)], bearer: Option<&str>) -> Result<reqwest::Client, Error> {
    let mut map = HeaderMap::new();
    for (name, value) in headers {
        let name = HeaderName::from_bytes(name.as_bytes()).map_err(failed)?;
        map.append(name, HeaderValue::from_str(value).map_err(failed)?);
    }
    if !map.contains_key(USER_AGENT) {
        map.insert(USER_AGENT, HeaderValue::from_static(DEFAULT_USER_AGENT));
    }
    if let Some(token) = bearer.filter(|token| !token.is_empty()) {
        let value = HeaderValue::from_str(&format!("Bearer {token}")).map_err(failed)?;
        map.append(AUTHORIZATION, value);
    }
    reqwest::Client::builder()
        .default_headers(map)
        .connect_timeout(CONNECT_TIMEOUT)
        .build()
        .map_err(failed)
}

fn content_length(headers: &HeaderMap, func: &str) -> Option<u64> {
    let value = headers.get(CONTENT_LENGTH)?;
    let length = value.to_str().ok().and_then(|value| value.parse().ok());
    if length.is_none() {
        log::emit(
            Level::Warn,
            format!(
                "{func}: invalid Content-Length header: {}",
                String::from_utf8_lossy(value.as_bytes())
            ),
        );
    }
    length
}

fn header_text(headers: &HeaderMap, name: &str) -> String {
    headers
        .get(name)
        .map(|value| String::from_utf8_lossy(value.as_bytes()).into_owned())
        .unwrap_or_default()
}

fn masked_url(url: &str) -> String {
    let Some(scheme_end) = url.find("://") else {
        return url.to_string();
    };
    let rest = &url[scheme_end + 3..];
    let authority = rest.split('/').next().unwrap_or_default();
    match authority.rfind('@') {
        Some(at) => format!("{}://***{}", &url[..scheme_end], &rest[at..]),
        None => url.to_string(),
    }
}

fn with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(suffix);
    PathBuf::from(name)
}

fn read_etag(path: &Path) -> String {
    let etag_path = with_suffix(path, ".etag");
    match fs::read(&etag_path) {
        Ok(data) => {
            let line = data.split(|&b| b == b'\n').next().unwrap_or_default();
            String::from_utf8_lossy(line).into_owned()
        }
        Err(_) => {
            log::emit(
                Level::Error,
                format!(
                    "read_etag: could not open .etag file for reading: {}",
                    etag_path.display()
                ),
            );
            String::new()
        }
    }
}

pub fn write_atomic(path: &Path, data: &[u8]) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp = with_suffix(path, ".tmp");
    fs::write(&tmp, data)?;
    fs::rename(tmp, path)
}

fn write_etag(path: &Path, etag: &str) {
    let etag_path = with_suffix(path, ".etag");
    match write_atomic(&etag_path, etag.as_bytes()) {
        Ok(()) => log::emit(
            Level::Debug,
            format!("write_etag: file etag saved: {}", etag_path.display()),
        ),
        Err(error) => log::emit(
            Level::Error,
            format!(
                "write_etag: unable to write file: {}: {error}",
                etag_path.display()
            ),
        ),
    }
}

async fn get_body(
    client: &reqwest::Client,
    url: &str,
    timeout: Duration,
    max_size: usize,
) -> Option<(u16, Vec<u8>)> {
    let mut response = tokio::time::timeout(timeout, client.get(url).send())
        .await
        .ok()?
        .ok()?;
    let status = response.status().as_u16();
    let mut body = Vec::new();
    while let Some(chunk) = tokio::time::timeout(timeout, response.chunk())
        .await
        .ok()?
        .ok()?
    {
        body.extend_from_slice(&chunk);
        if max_size != 0 && body.len() > max_size {
            return None;
        }
    }
    Some((status, body))
}

pub fn get_content(url: &str, params: &ContentParams) -> Result<(u16, Vec<u8>), Error> {
    let client = client(&params.headers, None)?;
    let timeout = params.timeout.unwrap_or(READ_TIMEOUT);
    runtime()?
        .block_on(get_body(&client, url, timeout, params.max_size))
        .ok_or_else(|| Error::Failed("error: cannot make GET request".into()))
}

fn pull_failed(reason: impl std::fmt::Display, status: i32) -> bool {
    log::emit(
        Level::Error,
        format!("common_pull_file: download failed: {reason} (status: {status})"),
    );
    false
}

async fn pull(
    client: &reqwest::Client,
    url: &str,
    tmp: &Path,
    supports_ranges: bool,
    progress: &mut Progress,
    callback: Option<&dyn Callback>,
) -> bool {
    let Ok(mut file) = OpenOptions::new().create(true).append(true).open(tmp) else {
        log::emit(
            Level::Error,
            format!(
                "common_pull_file: error opening local file for writing: {}",
                tmp.display()
            ),
        );
        return false;
    };
    let mut request = client.get(url);
    if supports_ranges && progress.downloaded > 0 {
        request = request.header(RANGE, format!("bytes={}-", progress.downloaded));
    }
    let mut response = match tokio::time::timeout(READ_TIMEOUT, request.send()).await {
        Ok(Ok(response)) => response,
        Ok(Err(error)) => return pull_failed(error, -1),
        Err(_) => return pull_failed("timed out", -1),
    };
    let status = response.status().as_u16();
    if progress.downloaded > 0 && status != 206 {
        log::emit(
            Level::Warn,
            format!(
                "common_pull_file: server did not respond with 206 Partial Content for a resume request. Status: {status}"
            ),
        );
        return false;
    }
    if progress.downloaded == 0 && status != 200 {
        log::emit(
            Level::Warn,
            format!("common_pull_file: download received non-successful status code: {status}"),
        );
        return false;
    }
    if progress.total == 0
        && let Some(length) = content_length(response.headers(), "common_pull_file")
    {
        progress.total = progress.downloaded + length;
    }
    let mut step = 0u64;
    loop {
        let chunk = match tokio::time::timeout(READ_TIMEOUT, response.chunk()).await {
            Ok(Ok(Some(chunk))) => chunk,
            Ok(Ok(None)) => return true,
            Ok(Err(error)) => return pull_failed(error, i32::from(status)),
            Err(_) => return pull_failed("timed out", i32::from(status)),
        };
        if file.write_all(&chunk).is_err() {
            log::emit(
                Level::Error,
                format!("common_pull_file: error writing to file: {}", tmp.display()),
            );
            return false;
        }
        progress.downloaded += chunk.len() as u64;
        step += chunk.len() as u64;
        if step >= progress.total / 1000 || progress.downloaded == progress.total {
            if let Some(callback) = callback {
                callback.on_update(progress);
                if callback.is_cancelled() {
                    return false;
                }
            }
            step = 0;
        }
    }
}

fn online(url: &str, path: &Path, opts: &Options, skip_etag: bool) -> Result<u16, Error> {
    let file_exists = path.exists();
    if file_exists && skip_etag {
        log::emit(
            Level::Debug,
            format!(
                "common_download_file_single_online: using cached file: {}",
                path.display()
            ),
        );
        return Ok(304);
    }

    let client = client(&opts.headers, opts.bearer_token.as_deref())?;
    let runtime = runtime()?;
    let last_etag = if file_exists {
        read_etag(path)
    } else {
        log::emit(
            Level::Debug,
            format!(
                "common_download_file_single_online: no previous model file found {}",
                path.display()
            ),
        );
        String::new()
    };

    let head = runtime.block_on(async {
        tokio::time::timeout(READ_TIMEOUT, client.head(url).send())
            .await
            .ok()?
            .ok()
    });
    let head = match head {
        Some(head) if head.status().is_success() => head,
        other => {
            if file_exists {
                return Ok(304);
            }
            return Err(match other {
                Some(head) => Error::Status(head.status().as_u16()),
                None => Error::Failed(format!("HEAD request to {url} failed")),
            });
        }
    };

    let status = head.status().as_u16();
    let etag = header_text(head.headers(), "etag");
    let supports_ranges = head
        .headers()
        .get("accept-ranges")
        .is_some_and(|value| value.as_bytes() != b"none");

    if file_exists {
        if etag.is_empty() {
            log::emit(
                Level::Debug,
                format!(
                    "common_download_file_single_online: using cached file (no server etag): {}",
                    path.display()
                ),
            );
            return Ok(304);
        }
        if !last_etag.is_empty() && last_etag == etag {
            log::emit(
                Level::Debug,
                format!(
                    "common_download_file_single_online: using cached file (same etag): {}",
                    path.display()
                ),
            );
            return Ok(304);
        }
        if fs::remove_file(path).is_err() {
            let message = format!(
                "common_download_file_single_online: unable to delete file: {}",
                path.display()
            );
            log::emit(Level::Error, &message);
            return Err(Error::Failed(message));
        }
    }

    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }

    let tmp = with_suffix(path, ".downloadInProgress");
    let mut progress = Progress {
        url: url.to_string(),
        total: content_length(head.headers(), "common_download_file_single_online").unwrap_or(0),
        ..Default::default()
    };
    if let Some(callback) = opts.callback {
        callback.on_start(&progress);
    }

    let mut success = false;
    let mut delay = RETRY_DELAY;
    for attempt in 0..MAX_ATTEMPTS {
        if opts
            .callback
            .is_some_and(|callback| callback.is_cancelled())
        {
            break;
        }
        if attempt > 0 {
            log::emit(
                Level::Warn,
                format!(
                    "common_download_file_single_online: retrying after {} seconds...",
                    delay.as_secs()
                ),
            );
            thread::sleep(delay);
            delay *= 2;
        }

        progress.downloaded = match fs::metadata(&tmp) {
            Ok(meta) if supports_ranges => meta.len(),
            Ok(_) => {
                if fs::remove_file(&tmp).is_err() {
                    log::emit(
                        Level::Error,
                        format!(
                            "common_download_file_single_online: unable to delete file: {}",
                            tmp.display()
                        ),
                    );
                    break;
                }
                0
            }
            Err(_) => 0,
        };

        log::emit(
            Level::Debug,
            format!(
                "common_download_file_single_online: downloading from {} to {} (etag:{etag})...",
                masked_url(url),
                tmp.display()
            ),
        );
        if runtime.block_on(pull(
            &client,
            url,
            &tmp,
            supports_ranges,
            &mut progress,
            opts.callback,
        )) {
            if fs::rename(&tmp, path).is_err() {
                log::emit(
                    Level::Error,
                    format!(
                        "common_download_file_single_online: unable to rename file: {} to {}",
                        tmp.display(),
                        path.display()
                    ),
                );
                break;
            }
            if !etag.is_empty() && !skip_etag {
                write_etag(path, &etag);
            }
            success = true;
            break;
        }
    }

    if let Some(callback) = opts.callback {
        callback.on_done(&progress, success);
    }
    if opts
        .callback
        .is_some_and(|callback| callback.is_cancelled())
        && tmp.exists()
        && fs::remove_file(&tmp).is_err()
    {
        log::emit(
            Level::Error,
            format!(
                "common_download_file_single_online: unable to delete temporary file: {}",
                tmp.display()
            ),
        );
    }
    if !success {
        let message = format!(
            "common_download_file_single_online: download failed after {MAX_ATTEMPTS} attempts"
        );
        log::emit(Level::Error, &message);
        return Err(Error::Failed(message));
    }
    Ok(status)
}

pub fn download_file(
    url: &str,
    path: &Path,
    opts: &Options,
    skip_etag: bool,
) -> Result<u16, Error> {
    if !opts.offline {
        return online(url, path, opts, skip_etag);
    }
    if !path.exists() {
        let message = format!(
            "common_download_file_single: required file is not available in cache (offline mode): {}",
            path.display()
        );
        log::emit(Level::Error, &message);
        return Err(Error::Failed(message));
    }
    log::emit(
        Level::Debug,
        format!(
            "common_download_file_single: using cached file (offline mode): {}",
            path.display()
        ),
    );
    if let Some(callback) = opts.callback {
        let progress = Progress {
            url: url.to_string(),
            cached: true,
            ..Default::default()
        };
        callback.on_start(&progress);
        callback.on_done(&progress, true);
    }
    Ok(304)
}
