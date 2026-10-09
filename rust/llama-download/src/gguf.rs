use std::sync::LazyLock;

use regex::Regex;

static SPLIT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^(.+)-([0-9]{5})-(?i:of)-([0-9]{5})$").unwrap());
static TAG: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[-.]([A-Za-z0-9_]+)$").unwrap());

const SIDECAR_MARKERS: [&str; 6] = ["mmproj", "imatrix", "mtp-", "eagle3-", "dflash-", "dspark-"];

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct SplitInfo {
    pub prefix: String,
    pub tag: String,
    pub index: u32,
    pub count: u32,
}

pub fn split_info(path: &str) -> SplitInfo {
    let Some(stem) = path.strip_suffix(".gguf") else {
        return SplitInfo::default();
    };
    let (prefix, index, count) = match SPLIT.captures(stem) {
        Some(c) => (
            c[1].to_string(),
            c[2].parse::<u32>().unwrap(),
            c[3].parse::<u32>().unwrap(),
        ),
        None => (stem.to_string(), 1, 1),
    };
    let tag = TAG
        .captures(&prefix)
        .map(|c| c[1].to_ascii_uppercase())
        .unwrap_or_default();
    SplitInfo {
        prefix,
        tag,
        index,
        count,
    }
}

pub fn bits_of_tag(tag: &str) -> i64 {
    let Some(pos) = tag.find(|c: char| c.is_ascii_digit()) else {
        return 0;
    };
    let digits: String = tag[pos..]
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits.parse().unwrap_or(i64::MAX)
}

pub fn quant_bits(path: &str) -> i64 {
    bits_of_tag(&split_info(path).tag)
}

pub fn is_model(path: &str) -> bool {
    if !path.ends_with(".gguf") {
        return false;
    }
    let name = path.rsplit('/').next().unwrap_or(path);
    SIDECAR_MARKERS.iter().all(|marker| !name.contains(marker))
}

pub fn split_files<S: AsRef<str>>(paths: &[S], file: usize) -> Vec<usize> {
    let split = split_info(paths[file].as_ref());
    if split.count <= 1 {
        return vec![file];
    }
    paths
        .iter()
        .enumerate()
        .filter(|(_, path)| {
            let other = split_info(path.as_ref());
            other.count == split.count && other.prefix == split.prefix
        })
        .map(|(i, _)| i)
        .collect()
}

pub fn all_parts(url: &str) -> Vec<String> {
    let split = split_info(url);
    if split.count <= 1 {
        return vec![url.to_string()];
    }
    (1..=split.count)
        .map(|i| format!("{}-{i:05}-of-{:05}.gguf", split.prefix, split.count))
        .collect()
}
