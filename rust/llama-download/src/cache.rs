use std::collections::HashSet;
use std::fs;
use std::path::{Component, Path, PathBuf};

use crate::gguf;
use crate::log::{self, Level};
use crate::repo::{self, InvalidRepo};
use crate::select;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CachedFile {
    pub repo_id: String,
    pub path: String,
    pub local_path: PathBuf,
}

#[derive(Debug, PartialEq, Eq)]
pub struct CachedModel {
    pub repo_id: String,
    pub tag: String,
}

fn non_empty(var: &impl Fn(&str) -> Option<String>, name: &str) -> Option<PathBuf> {
    var(name)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

pub fn cache_dir_from(
    var: impl Fn(&str) -> Option<String>,
    home: Option<PathBuf>,
) -> Option<PathBuf> {
    for name in ["LLAMA_CACHE", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"] {
        if let Some(dir) = non_empty(&var, name) {
            return Some(dir);
        }
    }
    if let Some(home_dir) = non_empty(&var, "HF_HOME") {
        return Some(home_dir.join("hub"));
    }
    if let Some(xdg) = non_empty(&var, "XDG_CACHE_HOME") {
        return Some(xdg.join("huggingface").join("hub"));
    }
    home.map(|home| home.join(".cache").join("huggingface").join("hub"))
}

pub fn cache_dir() -> Option<PathBuf> {
    cache_dir_from(|name| std::env::var(name).ok(), std::env::home_dir())
}

fn llama_cache_base(
    var: &impl Fn(&str) -> Option<String>,
    home: Option<PathBuf>,
) -> Option<PathBuf> {
    if cfg!(target_os = "windows") {
        non_empty(var, "LOCALAPPDATA")
    } else if cfg!(target_os = "macos") {
        home.map(|home| home.join("Library").join("Caches"))
    } else {
        non_empty(var, "XDG_CACHE_HOME").or_else(|| home.map(|home| home.join(".cache")))
    }
}

pub fn llama_cache_dir_from(
    var: impl Fn(&str) -> Option<String>,
    home: Option<PathBuf>,
) -> Option<PathBuf> {
    if let Some(dir) = non_empty(&var, "LLAMA_CACHE") {
        return Some(dir);
    }
    llama_cache_base(&var, home).map(|base| base.join("llama.cpp"))
}

pub fn llama_cache_dir() -> Option<PathBuf> {
    llama_cache_dir_from(|name| std::env::var(name).ok(), std::env::home_dir())
}

pub fn repo_path(root: &Path, repo_id: &str) -> PathBuf {
    root.join(repo_folder(repo_id))
}

pub fn is_valid_repo_id(repo_id: &str) -> bool {
    if repo_id.is_empty() || repo_id.len() > 256 {
        return false;
    }
    let mut slashes = 0;
    let mut special = true;
    for c in repo_id.bytes() {
        if c.is_ascii_alphanumeric() || c == b'_' {
            special = false;
        } else if matches!(c, b'/' | b'.' | b'-') {
            if special {
                return false;
            }
            slashes += usize::from(c == b'/');
            special = true;
        } else {
            return false;
        }
    }
    !special && slashes == 1
}

fn is_valid_commit(hash: &[u8]) -> bool {
    hash.len() == 40 && hash.iter().all(u8::is_ascii_hexdigit)
}

fn folder_repo(folder: &str) -> Option<String> {
    folder
        .strip_prefix("models--")
        .map(|rest| rest.replace("--", "/"))
}

pub(crate) fn repo_folder(repo_id: &str) -> String {
    format!("models--{}", repo_id.replace('/', "--"))
}

fn cached_ref(repo_path: &Path) -> Option<String> {
    let mut fallback = None;
    for entry in fs::read_dir(repo_path.join("refs")).ok()?.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Ok(data) = fs::read(&path) else {
            continue;
        };
        let line = data.split(|&b| b == b'\n').next().unwrap_or_default();
        if !is_valid_commit(line) {
            log::emit(
                Level::Warn,
                format!(
                    "get_cached_ref: skip invalid commit: {}",
                    String::from_utf8_lossy(line)
                ),
            );
            continue;
        }
        let commit = String::from_utf8_lossy(line).into_owned();
        if path.file_name().is_some_and(|name| name == "main") {
            return Some(commit);
        }
        fallback.get_or_insert(commit);
    }
    fallback
}

fn collect_files(base: &Path, dir: &Path, repo_id: &str, out: &mut Vec<CachedFile>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(kind) = entry.file_type() else {
            continue;
        };
        if kind.is_dir() {
            collect_files(base, &path, repo_id, out);
        } else if kind.is_symlink() || kind.is_file() {
            let Ok(relative) = path.strip_prefix(base) else {
                continue;
            };
            let rel = relative
                .components()
                .filter_map(|c| c.as_os_str().to_str())
                .collect::<Vec<_>>()
                .join("/");
            out.push(CachedFile {
                repo_id: repo_id.to_string(),
                path: rel,
                local_path: path,
            });
        }
    }
}

pub fn cached_files(root: &Path, repo_filter: Option<&str>) -> Vec<CachedFile> {
    let mut files = Vec::new();
    let Ok(entries) = fs::read_dir(root) else {
        return files;
    };
    if let Some(filter) = repo_filter
        && !is_valid_repo_id(filter)
    {
        log::emit(
            Level::Warn,
            format!("get_cached_files: invalid repository: {filter}"),
        );
        return files;
    }
    for entry in entries.flatten() {
        let repo_path = entry.path();
        let Some(folder) = entry.file_name().to_str().map(str::to_string) else {
            continue;
        };
        if !repo_path.is_dir() || !repo_path.join("snapshots").exists() {
            continue;
        }
        let Some(repo_id) = folder_repo(&folder).filter(|id| is_valid_repo_id(id)) else {
            continue;
        };
        if repo_filter.is_some_and(|filter| filter != repo_id) {
            continue;
        }
        let Some(commit) = cached_ref(&repo_path) else {
            continue;
        };
        let commit_path = repo_path.join("snapshots").join(&commit);
        if commit_path.is_dir() {
            collect_files(&commit_path, &commit_path, &repo_id, &mut files);
        }
    }
    files
}

pub fn cached_models(root: &Path) -> Vec<CachedModel> {
    const EXCLUDED: [&str; 5] = ["mmproj", "mtp-", "eagle3-", "dflash-", "dspark-"];
    let mut seen = HashSet::new();
    let mut models = Vec::new();
    for file in cached_files(root, None) {
        let split = gguf::split_info(&file.path);
        if split.index != 1
            || split.tag.is_empty()
            || EXCLUDED.iter().any(|marker| split.prefix.contains(marker))
        {
            continue;
        }
        if seen.insert((file.repo_id.clone(), split.tag.clone())) {
            models.push(CachedModel {
                repo_id: file.repo_id,
                tag: split.tag,
            });
        }
    }
    models
}

pub fn resolve_path(root: &Path, spec: &str, file: &str) -> Result<Option<PathBuf>, InvalidRepo> {
    let (repo_id, tag) = repo::split_repo_tag(spec)?;
    let files = cached_files(root, Some(repo_id));
    if files.is_empty() {
        return Ok(None);
    }
    if !file.is_empty() {
        return Ok(files
            .into_iter()
            .find(|f| f.path == file)
            .map(|f| f.local_path));
    }
    let paths: Vec<&str> = files.iter().map(|f| f.path.as_str()).collect();
    Ok(select::find_best_model(&paths, tag).map(|i| files[i].local_path.clone()))
}

pub(crate) fn normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => match out.components().next_back() {
                Some(Component::Normal(_)) => {
                    out.pop();
                }
                Some(Component::RootDir) => {}
                _ => out.push(".."),
            },
            other => out.push(other),
        }
    }
    out
}

fn blob_of(link: &Path) -> Option<PathBuf> {
    let target = fs::read_link(link).ok()?;
    Some(normalize(&link.parent()?.join(target)))
}

fn remove_repo(root: &Path, repo_id: &str) -> bool {
    if !is_valid_repo_id(repo_id) {
        log::emit(
            Level::Warn,
            format!("remove_cached_repo: invalid repository: {repo_id}"),
        );
        return false;
    }
    let path = root.join(repo_folder(repo_id));
    match fs::remove_dir_all(&path) {
        Ok(()) => true,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
        Err(error) => {
            log::emit(
                Level::Error,
                format!(
                    "remove_cached_repo: failed to remove repo cache {}: {error}",
                    path.display()
                ),
            );
            false
        }
    }
}

fn remove_logged(path: &Path, label: &str) {
    if let Err(error) = fs::remove_file(path)
        && error.kind() != std::io::ErrorKind::NotFound
    {
        log::emit(
            Level::Warn,
            format!(
                "common_download_remove: failed to remove {label}{}: {error}",
                path.display()
            ),
        );
    }
}

pub fn remove(root: &Path, spec: &str) -> Result<bool, InvalidRepo> {
    let (repo_id, tag) = repo::split_repo_tag(spec)?;
    if tag.is_empty() {
        return Ok(remove_repo(root, repo_id));
    }
    let tag = tag.to_ascii_uppercase();
    let files = cached_files(root, Some(repo_id));
    let to_remove: Vec<PathBuf> = files
        .iter()
        .filter(|f| gguf::split_info(&f.path).tag == tag)
        .map(|f| f.local_path.clone())
        .collect();
    if to_remove.is_empty() {
        return Ok(false);
    }

    let blobs: Vec<PathBuf> = to_remove.iter().filter_map(|p| blob_of(p)).collect();
    for path in &to_remove {
        remove_logged(path, "");
    }
    if blobs.is_empty() {
        return Ok(true);
    }

    let referenced: HashSet<PathBuf> = cached_files(root, Some(repo_id))
        .iter()
        .filter_map(|f| blob_of(&f.local_path))
        .collect();
    for blob in blobs {
        if !referenced.contains(&blob) {
            remove_logged(&blob, "blob ");
        }
    }
    Ok(true)
}
