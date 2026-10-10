use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::log::{self, Level};
use crate::remote::{self, ContentParams, Error, Options};

const REGISTRY: &str = "https://registry-1.docker.io/v2";
const MANIFEST_ACCEPT: &str = "application/vnd.docker.distribution.manifest.v2+json,application/vnd.oci.image.manifest.v1+json";

fn get_json(url: &str, headers: Vec<(String, String)>, what: &str) -> Result<Value, Error> {
    let params = ContentParams {
        headers,
        ..Default::default()
    };
    let (status, body) = remote::get_content(url, &params)?;
    if status != 200 {
        return Err(Error::Failed(format!(
            "Failed to get Docker {what}, HTTP code: {status}"
        )));
    }
    serde_json::from_slice(&body).map_err(|error| Error::Failed(error.to_string()))
}

fn token(repo: &str) -> Result<String, Error> {
    let url = format!(
        "https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repo}:pull"
    );
    let json = get_json(&url, Vec::new(), "registry token")?;
    json.get("token")
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| Error::Failed("Docker registry token response missing 'token' field".into()))
}

fn gguf_digest(manifest: &Value) -> Result<&str, Error> {
    let layers = manifest.get("layers").and_then(Value::as_array);
    for layer in layers.into_iter().flatten() {
        let Some(media_type) = layer.get("mediaType").and_then(Value::as_str) else {
            continue;
        };
        if media_type.contains("gguf") {
            return layer
                .get("digest")
                .and_then(Value::as_str)
                .ok_or_else(|| Error::Failed("Docker manifest layer has no digest".into()));
        }
    }
    Err(Error::Failed(
        "No GGUF layer found in Docker manifest".into(),
    ))
}

fn normalize_digest(digest: &str) -> Result<String, Error> {
    digest
        .strip_prefix("sha256:")
        .filter(|hex| hex.len() == 64 && hex.bytes().all(|b| b.is_ascii_hexdigit()))
        .map(|hex| format!("sha256:{}", hex.to_ascii_lowercase()))
        .ok_or_else(|| {
            Error::Failed(format!(
                "Invalid OCI digest format received in manifest: {digest}"
            ))
        })
}

pub fn resolve_model(docker: &str, cache_dir: &Path) -> Result<PathBuf, Error> {
    let result = download_model(docker, cache_dir);
    if let Err(error) = &result {
        log::emit(
            Level::Error,
            format!("common_docker_resolve_model: Docker Model download failed: {error}"),
        );
    }
    result
}

fn download_model(docker: &str, cache_dir: &Path) -> Result<PathBuf, Error> {
    let (mut repo, tag) = match docker.split_once(':') {
        Some((repo, tag)) => (repo.to_string(), tag.to_string()),
        None => (docker.to_string(), "latest".to_string()),
    };
    if !docker.contains('/') {
        repo.insert_str(0, "ai/");
    }
    log::emit(
        Level::Info,
        format!("common_docker_resolve_model: Downloading Docker Model: {repo}:{tag}"),
    );
    let file_name = format!("{}_{tag}.gguf", repo.replace('/', "_"));
    if file_name.contains(['/', '\\']) {
        return Err(Error::Failed(format!("invalid Docker model tag: {tag}")));
    }

    let token = token(&repo)?;
    let prefix = format!("{REGISTRY}/{repo}");
    let manifest_headers = vec![
        ("Authorization".to_string(), format!("Bearer {token}")),
        ("Accept".to_string(), MANIFEST_ACCEPT.to_string()),
    ];
    let manifest = get_json(
        &format!("{prefix}/manifests/{tag}"),
        manifest_headers,
        "manifest",
    )?;
    let digest = normalize_digest(gguf_digest(&manifest)?)?;
    log::emit(
        Level::Debug,
        format!("common_docker_resolve_model: Using validated digest: {digest}"),
    );

    fs::create_dir_all(cache_dir).map_err(|error| {
        Error::Failed(format!(
            "failed to create cache directory: {}: {error}",
            cache_dir.display()
        ))
    })?;
    let local_path = cache_dir.join(file_name);
    let opts = Options {
        bearer_token: Some(token),
        ..Default::default()
    };
    remote::download_file(
        &format!("{prefix}/blobs/{digest}"),
        &local_path,
        &opts,
        false,
    )
    .map_err(|_| Error::Failed("Failed to download Docker Model".into()))?;
    log::emit(
        Level::Info,
        format!(
            "common_docker_resolve_model: Downloaded Docker Model to: {}",
            local_path.display()
        ),
    );
    Ok(local_path)
}
