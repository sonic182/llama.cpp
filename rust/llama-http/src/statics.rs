use std::{
    path::{Path, PathBuf},
    time::UNIX_EPOCH,
};

use bytes::Bytes;

use crate::codec;

pub struct StaticDir {
    root: PathBuf,
    canonical_root: PathBuf,
    mount: String,
}

pub enum Outcome {
    Forbidden,
    Redirect(String),
    NotModified { etag: String },
    File(StaticFile),
}

pub struct StaticFile {
    pub content_type: &'static str,
    pub etag: String,
    pub data: Bytes,
}

impl StaticDir {
    pub fn new(root: &str, prefix: &str) -> std::io::Result<Self> {
        let root = PathBuf::from(root);
        let canonical_root = root.canonicalize()?;
        if !canonical_root.is_dir() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "not a directory",
            ));
        }
        Ok(Self {
            root,
            canonical_root,
            mount: format!("{prefix}/"),
        })
    }

    pub async fn lookup(
        &self,
        request_path: &str,
        if_none_match: Option<&[u8]>,
    ) -> Option<Outcome> {
        let relative = request_path.strip_prefix(&self.mount)?;
        let decoded = codec::decode(relative.as_bytes(), false);
        let sub_path = format!("/{}", String::from_utf8_lossy(&decoded));
        if !is_valid_path(&sub_path) {
            return None;
        }

        let mut path = self.root.join(sub_path.trim_start_matches('/'));
        if sub_path.ends_with('/') {
            path.push("index.html");
        }

        let resolved = tokio::fs::canonicalize(&path).await.ok()?;
        if !resolved.starts_with(&self.canonical_root) {
            return Some(Outcome::Forbidden);
        }

        let metadata = tokio::fs::metadata(&resolved).await.ok()?;
        if metadata.is_dir() {
            return Some(Outcome::Redirect(format!("{request_path}/")));
        }
        if !metadata.is_file() {
            return None;
        }

        let mtime = metadata
            .modified()
            .ok()
            .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
            .map_or(0, |elapsed| elapsed.as_secs());
        let etag = format!("W/\"{mtime:x}-{:x}\"", metadata.len());
        if if_none_match.is_some_and(|candidates| matches_etag(candidates, &etag)) {
            return Some(Outcome::NotModified { etag });
        }

        let data = tokio::fs::read(&resolved).await.ok()?;
        Some(Outcome::File(StaticFile {
            content_type: content_type(&resolved),
            etag,
            data: Bytes::from(data),
        }))
    }
}

fn matches_etag(candidates: &[u8], etag: &str) -> bool {
    candidates
        .split(|&byte| byte == b',')
        .map(<[u8]>::trim_ascii)
        .any(|candidate| candidate == b"*" || candidate == etag.as_bytes())
}

fn is_valid_path(path: &str) -> bool {
    if path.contains(['\0', '\\']) {
        return false;
    }
    let mut level = 0usize;
    for component in path.split('/').filter(|component| !component.is_empty()) {
        match component {
            "." => {}
            ".." => match level.checked_sub(1) {
                Some(next) => level = next,
                None => return false,
            },
            _ => level += 1,
        }
    }
    true
}

fn content_type(path: &Path) -> &'static str {
    let extension = path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_ascii_lowercase)
        .unwrap_or_default();
    match extension.as_str() {
        "css" => "text/css",
        "csv" => "text/csv",
        "htm" | "html" => "text/html",
        "js" | "mjs" => "text/javascript",
        "txt" => "text/plain",
        "vtt" => "text/vtt",
        "apng" => "image/apng",
        "avif" => "image/avif",
        "bmp" => "image/bmp",
        "gif" => "image/gif",
        "png" => "image/png",
        "svg" => "image/svg+xml",
        "webp" => "image/webp",
        "ico" => "image/x-icon",
        "tif" | "tiff" => "image/tiff",
        "jpeg" | "jpg" => "image/jpeg",
        "mp4" => "video/mp4",
        "mpeg" => "video/mpeg",
        "webm" => "video/webm",
        "mp3" | "mpga" => "audio/mpeg",
        "weba" => "audio/webm",
        "wav" => "audio/wave",
        "otf" => "font/otf",
        "ttf" => "font/ttf",
        "woff" => "font/woff",
        "woff2" => "font/woff2",
        "7z" => "application/x-7z-compressed",
        "atom" => "application/atom+xml",
        "pdf" => "application/pdf",
        "json" => "application/json",
        "rss" => "application/rss+xml",
        "tar" => "application/x-tar",
        "xht" | "xhtml" => "application/xhtml+xml",
        "xslt" => "application/xslt+xml",
        "xml" => "application/xml",
        "gz" => "application/gzip",
        "zip" => "application/zip",
        "wasm" => "application/wasm",
        _ => "application/octet-stream",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_traversal_and_backslashes() {
        assert!(is_valid_path("/a/b/../c"));
        assert!(is_valid_path("/./a"));
        assert!(!is_valid_path("/../a"));
        assert!(!is_valid_path("/a/../../b"));
        assert!(!is_valid_path("/a\\b"));
    }

    #[test]
    fn etag_list_matching() {
        assert!(matches_etag(b"W/\"1-2\", W/\"3-4\"", "W/\"3-4\""));
        assert!(matches_etag(b"*", "W/\"3-4\""));
        assert!(!matches_etag(b"W/\"1-2\"", "W/\"3-4\""));
    }
}
