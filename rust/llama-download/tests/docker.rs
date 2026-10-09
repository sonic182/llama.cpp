use std::fs;

use llama_download::docker;

const MODEL: &str = "ai/smollm2:135M-Q2_K";
const FILE: &str = "ai_smollm2_135M-Q2_K.gguf";
const SIZE: u64 = 88202112;

#[test]
fn docker_model_resolves_to_cache_and_revalidates() {
    let tmp = tempfile::tempdir().unwrap();
    let path = docker::resolve_model(MODEL, tmp.path()).unwrap();
    assert_eq!(path, tmp.path().join(FILE));
    assert_eq!(fs::metadata(&path).unwrap().len(), SIZE);
    let modified = fs::metadata(&path).unwrap().modified().unwrap();

    assert_eq!(docker::resolve_model(MODEL, tmp.path()).unwrap(), path);
    assert_eq!(fs::metadata(&path).unwrap().modified().unwrap(), modified);
}
