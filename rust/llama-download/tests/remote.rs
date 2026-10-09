use std::fs;

use llama_download::remote::{self, Options};

const URL: &str =
    "https://huggingface.co/ggml-org/test-model-stories260K/resolve/main/stories260K-f32.gguf";
const SIZE: usize = 1185376;

#[test]
fn plain_url_download_resumes_and_revalidates_etag() {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    runtime.block_on(async {
        tokio::task::spawn_blocking(download_flow).await.unwrap();
    });
}

fn download_flow() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let opts = Options::default();

    let full = root.join("full.gguf");
    assert_eq!(remote::download_file(URL, &full, &opts, false), Ok(200));
    let data = fs::read(&full).unwrap();
    assert_eq!(data.len(), SIZE);
    assert!(
        fs::read_to_string(root.join("full.gguf.etag"))
            .unwrap()
            .starts_with('"')
    );
    assert_eq!(remote::download_file(URL, &full, &opts, false), Ok(304));

    let resumed = root.join("resumed.gguf");
    fs::write(
        root.join("resumed.gguf.downloadInProgress"),
        &data[..100_000],
    )
    .unwrap();
    assert_eq!(remote::download_file(URL, &resumed, &opts, false), Ok(200));
    assert_eq!(fs::read(&resumed).unwrap(), data);

    let offline = Options {
        offline: true,
        ..Default::default()
    };
    assert_eq!(remote::download_file(URL, &full, &offline, false), Ok(304));
    assert!(remote::download_file(URL, &root.join("missing.gguf"), &offline, false).is_err());
}
