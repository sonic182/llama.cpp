use std::fs;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::{Duration, Instant};

use llama_download::remote::{self, ContentParams, Options};

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

fn read_request(stream: &mut TcpStream) -> String {
    let mut request = Vec::new();
    let mut buf = [0u8; 1024];
    while !request.ends_with(b"\r\n\r\n") {
        match stream.read(&mut buf) {
            Ok(0) | Err(_) => break,
            Ok(n) => request.extend_from_slice(&buf[..n]),
        }
    }
    String::from_utf8_lossy(&request).to_ascii_lowercase()
}

fn serve(handler: impl Fn(&str, &mut TcpStream) + Send + 'static) -> (String, Receiver<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else {
                continue;
            };
            let request = read_request(&mut stream);
            handler(&request, &mut stream);
            let _ = tx.send(request);
        }
    });
    (base, rx)
}

fn hello(request: &str, stream: &mut TcpStream) {
    let head = "HTTP/1.1 200 OK\r\nContent-Length: 5\r\nETag: \"x\"\r\nConnection: close\r\n\r\n";
    let body = if request.starts_with("head") {
        ""
    } else {
        "hello"
    };
    let _ = stream.write_all(format!("{head}{body}").as_bytes());
}

#[test]
fn slow_response_with_steady_progress_is_not_a_total_timeout() {
    let (base, _) = serve(|request, stream| {
        if request.starts_with("head") {
            return hello(request, stream);
        }
        for part in [
            &b"HTTP/1.1 200 OK\r\n"[..],
            b"Content-Length: 5\r\n",
            b"ETag: \"x\"\r\nConnection: close\r\n\r\nhello",
        ] {
            thread::sleep(Duration::from_secs(2));
            if stream.write_all(part).is_err() {
                break;
            }
        }
    });
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("model.gguf");
    let url = format!("{base}/model.gguf");
    assert_eq!(
        remote::download_file(&url, &path, &Options::default(), false),
        Ok(200)
    );
    assert_eq!(fs::read(&path).unwrap(), b"hello");
}

#[test]
fn stall_longer_than_five_seconds_does_not_fail_the_download() {
    let (base, _) = serve(|request, stream| {
        thread::sleep(Duration::from_secs(6));
        hello(request, stream);
    });
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("m.gguf");
    let url = format!("{base}/model.gguf");
    assert_eq!(
        remote::download_file(&url, &path, &Options::default(), false),
        Ok(200)
    );
    assert_eq!(fs::read(&path).unwrap(), b"hello");
}

#[test]
fn silent_server_fails_after_the_requested_timeout() {
    let (base, _) = serve(|_, _| thread::sleep(Duration::from_secs(60)));
    let start = Instant::now();
    let params = ContentParams {
        timeout: Some(Duration::from_secs(1)),
        ..ContentParams::default()
    };
    assert!(remote::get_content(&format!("{base}/x"), &params).is_err());
    let elapsed = start.elapsed();
    assert!(
        elapsed >= Duration::from_secs(1) && elapsed < Duration::from_secs(10),
        "{elapsed:?}"
    );
}

#[test]
fn url_credentials_are_sent_as_basic_auth() {
    let (base, requests) = serve(hello);
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("model.gguf");
    let url = format!("{}/model.gguf", base.replace("http://", "http://user:pw@"));
    assert_eq!(
        remote::download_file(&url, &path, &Options::default(), false),
        Ok(200)
    );
    let requests: Vec<String> = requests.try_iter().collect();
    assert_eq!(requests.len(), 2);
    assert!(
        requests
            .iter()
            .all(|r| r.contains("\r\nauthorization: basic dxnlcjpwdw==\r\n"))
    );
}

#[test]
fn cross_origin_redirect_drops_the_bearer_token() {
    let (target, target_requests) = serve(hello);
    let (origin, origin_requests) = serve(move |_, stream| {
        let response = format!(
            "HTTP/1.1 302 Found\r\nLocation: {target}/blob\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
        );
        let _ = stream.write_all(response.as_bytes());
    });
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("model.gguf");
    let opts = Options {
        bearer_token: Some("secret".to_string()),
        ..Default::default()
    };
    let url = format!("{origin}/model.gguf");
    assert_eq!(remote::download_file(&url, &path, &opts, false), Ok(200));
    assert_eq!(fs::read(&path).unwrap(), b"hello");

    let origin_requests: Vec<String> = origin_requests.try_iter().collect();
    let target_requests: Vec<String> = target_requests.try_iter().collect();
    assert_eq!(origin_requests.len(), 2);
    assert_eq!(target_requests.len(), 2);
    assert!(
        origin_requests
            .iter()
            .all(|r| r.contains("authorization: bearer secret"))
    );
    assert!(target_requests.iter().all(|r| !r.contains("authorization")));
    assert!(target_requests.iter().all(|r| r.contains("/blob ")));
}
