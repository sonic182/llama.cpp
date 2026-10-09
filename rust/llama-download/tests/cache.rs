use std::fs;
use std::os::unix::fs::symlink;
use std::path::{Path, PathBuf};

use llama_download::cache;

const COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";
const OTHER: &str = "fedcba9876543210fedcba9876543210fedcba98";

fn write(path: PathBuf, data: &str) {
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, data).unwrap();
}

fn link(target: &str, path: PathBuf) {
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    symlink(target, path).unwrap();
}

fn build_tree(root: &Path) {
    let model = root.join("models--org--model");
    write(model.join("refs/main"), COMMIT);
    for blob in ["bl1", "bl2", "bl3", "bl4", "bl5"] {
        write(model.join("blobs").join(blob), blob);
    }
    let snap = model.join("snapshots").join(COMMIT);
    link("../../blobs/bl1", snap.join("Model-Q4_K_M.gguf"));
    link("../../blobs/bl2", snap.join("Model-Q8_0.gguf"));
    link("../../blobs/bl3", snap.join("mmproj-f16.gguf"));
    write(snap.join("README.md"), "readme");
    link("../../../blobs/bl4", snap.join("sub/x-00001-of-00002.gguf"));
    link("../../../blobs/bl5", snap.join("sub/x-00002-of-00002.gguf"));

    let nofallback = root.join("models--org--nofallback");
    write(nofallback.join("refs/dev"), OTHER);
    write(nofallback.join("blobs/bl6"), "bl6");
    link(
        "../../blobs/bl6",
        nofallback
            .join("snapshots")
            .join(OTHER)
            .join("Model-IQ2_XS.gguf"),
    );

    let shared = root.join("models--org--shared");
    write(shared.join("refs/main"), COMMIT);
    write(shared.join("blobs/bl7"), "bl7");
    let snap = shared.join("snapshots").join(COMMIT);
    link("../../blobs/bl7", snap.join("Model-Q8_0.gguf"));
    link("../../blobs/bl7", snap.join("Model-F16.gguf"));

    let bad = root.join("models--bad--");
    write(bad.join("refs/main"), COMMIT);
    link(
        "../../../blobs/bl1",
        bad.join("snapshots").join(COMMIT).join("Model-Q4_K_M.gguf"),
    );
    fs::create_dir_all(root.join("not-a-repo/snapshots")).unwrap();
}

fn run_op(root: &Path, op: &str, a: &str, b: &str) -> Vec<String> {
    match op {
        "files" => {
            let filter = (!a.is_empty()).then_some(a);
            let mut out: Vec<String> = cache::cached_files(root, filter)
                .iter()
                .map(|f| {
                    format!(
                        "files\t{}\t{}\t{}",
                        f.repo_id,
                        f.path,
                        f.local_path.display()
                    )
                })
                .collect();
            out.sort();
            out
        }
        "models" => {
            let mut out: Vec<String> = cache::cached_models(root)
                .iter()
                .map(|m| format!("models\t{}\t{}", m.repo_id, m.tag))
                .collect();
            out.sort();
            out
        }
        "resolve" => vec![match cache::resolve_path(root, a, b) {
            Ok(Some(path)) => format!("resolve\t{}", path.display()),
            Ok(None) => "resolve\t<none>".to_string(),
            Err(_) => "resolve\terror".to_string(),
        }],
        "remove" => vec![match cache::remove(root, a) {
            Ok(removed) => format!("remove\t{removed}"),
            Err(_) => "remove\terror".to_string(),
        }],
        other => panic!("unknown op {other}"),
    }
}

fn dump_tree(root: &Path) -> Vec<String> {
    fn collect(root: &Path, dir: &Path, lines: &mut Vec<String>) {
        for entry in fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            let meta = fs::symlink_metadata(&path).unwrap();
            let kind = if meta.is_symlink() {
                'l'
            } else if meta.is_dir() {
                'd'
            } else {
                'f'
            };
            let target = if meta.is_symlink() {
                fs::read_link(&path).unwrap().display().to_string()
            } else {
                String::new()
            };
            let rel = path.strip_prefix(root).unwrap().display();
            lines.push(format!("tree {kind} ./{rel} -> {target}"));
            if kind == 'd' {
                collect(root, &path, lines);
            }
        }
    }
    let mut lines = vec!["tree d . -> ".to_string()];
    collect(root, root, &mut lines);
    lines.sort();
    lines
}

#[test]
fn cache_ops_match_common_download_cpp() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    build_tree(root);

    let mut out = String::new();
    for line in include_str!("golden/cache_ops.txt").lines() {
        let mut parts = line.split('\t');
        let op = parts.next().unwrap();
        let a = parts.next().unwrap_or("");
        let b = parts.next().unwrap_or("");
        for text in run_op(root, op, a, b).into_iter().chain(dump_tree(root)) {
            out.push_str(&text);
            out.push('\n');
        }
    }
    let out = out.replace(root.to_str().unwrap(), "ROOT");
    assert_eq!(out, include_str!("golden/cache.txt"));
}
