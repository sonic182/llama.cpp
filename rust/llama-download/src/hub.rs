use std::fs;
use std::io;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use serde_json::{Value, json};

use crate::cache::{self, normalize};
use crate::log::{self, Level};
use crate::remote::{self, ContentParams, Error, write_atomic};
use crate::repo::{self, InvalidRepo};
use crate::select::{self, SelectError, Wanted};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HfFile {
    pub path: String,
    pub url: String,
    pub local_path: String,
    pub final_path: String,
    pub oid: String,
    pub repo_id: String,
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct Plan {
    pub preset: Option<HfFile>,
    pub primary: Option<HfFile>,
    pub model_files: Vec<HfFile>,
    pub mmproj: Option<HfFile>,
    pub mtp: Option<HfFile>,
    pub eagle3: Option<HfFile>,
    pub dflash: Option<HfFile>,
    pub dspark: Option<HfFile>,
}

pub fn to_json(file: &HfFile) -> Value {
    json!({
        "path": file.path,
        "url": file.url,
        "local_path": file.local_path,
        "final_path": file.final_path,
        "oid": file.oid,
        "repo_id": file.repo_id,
    })
}

pub fn endpoint() -> String {
    let env = |name: &str| std::env::var(name).ok().filter(|value| !value.is_empty());
    let mut endpoint = env("MODEL_ENDPOINT")
        .or_else(|| env("HF_ENDPOINT"))
        .unwrap_or_else(|| "https://huggingface.co/".to_string());
    if !endpoint.ends_with('/') {
        endpoint.push('/');
    }
    endpoint
}

fn valid_token(token: &str) -> bool {
    token.len() >= 37
        && token.len() <= 256
        && token.starts_with("hf_")
        && token[3..].bytes().all(|b| b.is_ascii_alphanumeric())
}

fn is_hex(value: &str, len: usize) -> bool {
    value.len() == len && value.bytes().all(|b| b.is_ascii_hexdigit())
}

fn valid_subpath(base: &Path, sub: &str) -> bool {
    let sub = Path::new(sub);
    if sub.is_absolute() {
        return false;
    }
    let Ok(base) = std::path::absolute(base) else {
        return false;
    };
    normalize(&base.join(sub)).starts_with(normalize(&base))
}

fn api_get(url: &str, token: &str) -> Result<Value, Error> {
    let mut headers = vec![("Accept".to_string(), "application/json".to_string())];
    if valid_token(token) {
        headers.push(("Authorization".to_string(), format!("Bearer {token}")));
    } else if !token.is_empty() {
        log::emit(
            Level::Warn,
            "api_get: invalid token, authentication disabled",
        );
    }
    let params = ContentParams {
        headers,
        ..Default::default()
    };
    let (status, body) = remote::get_content(url, &params)?;
    if status == 200 {
        return serde_json::from_slice(&body).map_err(|error| Error::Failed(error.to_string()));
    }
    let message = serde_json::from_slice::<Value>(&body)
        .ok()
        .and_then(|value| {
            value
                .get("error")
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .unwrap_or_else(|| String::from_utf8_lossy(&body).into_owned());
    Err(Error::Failed(format!("GET failed ({status}): {message}")))
}

fn repo_commit(root: &Path, repo: &str, token: &str) -> Option<String> {
    let json = match api_get(&format!("{}api/models/{repo}/refs", endpoint()), token) {
        Ok(json) => json,
        Err(error) => {
            log::emit(Level::Error, format!("get_repo_commit: error: {error}"));
            return None;
        }
    };
    let Some(branches) = json.get("branches").and_then(Value::as_array) else {
        log::emit(
            Level::Warn,
            format!("get_repo_commit: missing 'branches' for '{repo}'"),
        );
        return None;
    };
    let refs = cache::repo_path(root, repo).join("refs");

    let mut chosen: Option<(String, String)> = None;
    for branch in branches {
        let Some(name) = branch.get("name").and_then(Value::as_str) else {
            continue;
        };
        let Some(commit) = branch.get("targetCommit").and_then(Value::as_str) else {
            continue;
        };
        if !valid_subpath(&refs, name) {
            log::emit(
                Level::Warn,
                format!("get_repo_commit: skip invalid branch: {name}"),
            );
            continue;
        }
        if !is_hex(commit, 40) {
            log::emit(
                Level::Warn,
                format!("get_repo_commit: skip invalid commit: {commit}"),
            );
            continue;
        }
        if name == "main" {
            chosen = Some((name.to_string(), commit.to_string()));
            break;
        }
        if chosen.is_none() {
            chosen = Some((name.to_string(), commit.to_string()));
        }
    }

    let Some((name, commit)) = chosen else {
        log::emit(
            Level::Warn,
            format!("get_repo_commit: no valid branch for '{repo}'"),
        );
        return None;
    };
    if let Err(error) = write_atomic(&refs.join(&name), commit.as_bytes()) {
        log::emit(Level::Error, format!("get_repo_commit: error: {error}"));
        return None;
    }
    Some(commit)
}

pub fn repo_files(root: &Path, repo: &str, token: &str) -> Vec<HfFile> {
    if !cache::is_valid_repo_id(repo) {
        log::emit(
            Level::Warn,
            format!("get_repo_files: invalid repository: {repo}"),
        );
        return Vec::new();
    }
    let Some(commit) = repo_commit(root, repo, token) else {
        log::emit(
            Level::Warn,
            format!("get_repo_files: failed to resolve commit for {repo}"),
        );
        return Vec::new();
    };

    let endpoint = endpoint();
    let json = match api_get(
        &format!("{endpoint}api/models/{repo}/tree/{commit}?recursive=true"),
        token,
    ) {
        Ok(json) => json,
        Err(error) => {
            log::emit(Level::Error, format!("get_repo_files: error: {error}"));
            return Vec::new();
        }
    };
    let Some(items) = json.as_array() else {
        log::emit(
            Level::Warn,
            format!("get_repo_files: response is not an array for '{repo}'"),
        );
        return Vec::new();
    };

    let repo_path = cache::repo_path(root, repo);
    let blobs = repo_path.join("blobs");
    let commit_path = repo_path.join("snapshots").join(&commit);
    let mut files = Vec::new();
    for item in items {
        if item.get("type").and_then(Value::as_str) != Some("file") {
            continue;
        }
        let Some(path) = item.get("path").and_then(Value::as_str) else {
            continue;
        };
        if !valid_subpath(&commit_path, path) {
            log::emit(
                Level::Warn,
                format!("get_repo_files: skip invalid path: {path}"),
            );
            continue;
        }

        let oid = match item.get("lfs").filter(|lfs| lfs.is_object()) {
            Some(lfs) => lfs.get("oid").and_then(Value::as_str).unwrap_or_default(),
            None => item.get("oid").and_then(Value::as_str).unwrap_or_default(),
        };
        if !oid.is_empty() && !is_hex(oid, 40) && !is_hex(oid, 64) {
            log::emit(
                Level::Warn,
                format!("get_repo_files: skip invalid oid: {oid}"),
            );
            continue;
        }

        let final_path = commit_path.join(path);
        let local_path = if !oid.is_empty() && !final_path.exists() {
            blobs.join(oid)
        } else {
            final_path.clone()
        };
        files.push(HfFile {
            path: path.to_string(),
            url: format!("{endpoint}{repo}/resolve/{commit}/{path}"),
            local_path: local_path.to_string_lossy().into_owned(),
            final_path: final_path.to_string_lossy().into_owned(),
            oid: oid.to_string(),
            repo_id: repo.to_string(),
        });
    }
    files
}

pub fn cached_files(root: &Path, repo: Option<&str>) -> Vec<HfFile> {
    cache::cached_files(root, repo)
        .into_iter()
        .map(|file| HfFile {
            path: file.path,
            local_path: file.local_path.to_string_lossy().into_owned(),
            final_path: file.local_path.to_string_lossy().into_owned(),
            repo_id: file.repo_id,
            ..Default::default()
        })
        .collect()
}

fn list_available_gguf_files(files: &[HfFile]) {
    log::emit(Level::Info, "Available GGUF files:");
    for file in files.iter().filter(|file| file.path.ends_with(".gguf")) {
        log::emit(Level::Info, format!(" - {}", file.path));
    }
}

pub fn plan(
    root: &Path,
    spec: &str,
    hf_file: &str,
    offline: bool,
    token: &str,
    wanted: Wanted,
) -> Result<Plan, InvalidRepo> {
    let (repo, tag) = repo::split_repo_tag(spec)?;

    let mut all = if offline {
        Vec::new()
    } else {
        repo_files(root, repo, token)
    };
    if all.is_empty() {
        all = cached_files(root, Some(repo));
    }
    if all.is_empty() {
        return Ok(Plan::default());
    }

    let paths: Vec<&str> = all.iter().map(|file| file.path.as_str()).collect();
    let selection = match select::select_hf_plan(&paths, hf_file, tag, wanted) {
        Ok(selection) => selection,
        Err(error) => {
            let message = match error {
                SelectError::HfFileNotFound => format!("file '{hf_file}' not found in repository"),
                SelectError::NoGgufFiles => format!("no GGUF files found in repository {repo}"),
            };
            log::emit(
                Level::Error,
                format!("common_download_get_hf_plan: {message}"),
            );
            list_available_gguf_files(&all);
            return Ok(Plan::default());
        }
    };

    let pick = |index: Option<usize>| index.map(|i| all[i].clone());
    Ok(Plan {
        preset: pick(selection.preset),
        primary: pick(selection.primary),
        model_files: selection
            .model_files
            .iter()
            .map(|&i| all[i].clone())
            .collect(),
        mmproj: pick(selection.mmproj),
        mtp: pick(selection.mtp),
        eagle3: pick(selection.eagle3),
        dflash: pick(selection.dflash),
        dspark: pick(selection.dspark),
    })
}

fn relative(target: &Path, base: &Path) -> PathBuf {
    let target: Vec<Component> = target.components().collect();
    let base: Vec<Component> = base.components().collect();
    let common = target.iter().zip(&base).take_while(|(a, b)| a == b).count();
    let mut out = PathBuf::new();
    for _ in common..base.len() {
        out.push("..");
    }
    for component in &target[common..] {
        out.push(component);
    }
    out
}

fn symlink_file(target: &Path, link: &Path) -> io::Result<()> {
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(target, link)
    }
    #[cfg(windows)]
    {
        std::os::windows::fs::symlink_file(target, link)
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = (target, link);
        Err(io::Error::other("symlinks are not supported"))
    }
}

static SYMLINKS_DISABLED: AtomicBool = AtomicBool::new(false);

pub fn finalize(file: &HfFile) -> String {
    let local = Path::new(&file.local_path);
    let fin = Path::new(&file.final_path);
    if local == fin || fin.exists() || !local.exists() {
        return file.final_path.clone();
    }
    let Some(parent) = fin.parent() else {
        return file.final_path.clone();
    };
    let _ = fs::create_dir_all(parent);
    if !SYMLINKS_DISABLED.load(Ordering::Relaxed) {
        match symlink_file(&relative(local, parent), fin) {
            Ok(()) => return file.final_path.clone(),
            Err(error) => {
                if !SYMLINKS_DISABLED.swap(true, Ordering::Relaxed) {
                    log::emit(
                        Level::Warn,
                        format!("finalize_file: failed to create symlink: {error}"),
                    );
                    log::emit(Level::Warn, "finalize_file: switching to degraded mode");
                }
            }
        }
    }
    if let Err(error) = fs::rename(local, fin) {
        log::emit(
            Level::Warn,
            format!("finalize_file: failed to move file to snapshots: {error}"),
        );
        if let Err(error) = fs::copy(local, fin) {
            log::emit(
                Level::Error,
                format!("finalize_file: failed to copy file to snapshots: {error}"),
            );
        }
    }
    file.final_path.clone()
}
