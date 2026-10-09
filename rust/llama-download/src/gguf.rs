const SIDECAR_MARKERS: [&str; 6] = ["mmproj", "imatrix", "mtp-", "eagle3-", "dflash-", "dspark-"];

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct SplitInfo {
    pub prefix: String,
    pub tag: String,
    pub index: u32,
    pub count: u32,
}

fn five_digits(bytes: &[u8]) -> Option<u32> {
    if bytes.len() == 5 && bytes.iter().all(u8::is_ascii_digit) {
        std::str::from_utf8(bytes).ok()?.parse().ok()
    } else {
        None
    }
}

fn split_suffix(stem: &str) -> Option<(&str, u32, u32)> {
    const SUFFIX: usize = "-00001-of-00002".len();
    let cut = stem.len().checked_sub(SUFFIX)?;
    let suffix = &stem.as_bytes()[cut..];
    if suffix[0] != b'-'
        || suffix[6] != b'-'
        || suffix[9] != b'-'
        || !suffix[7..9].eq_ignore_ascii_case(b"of")
    {
        return None;
    }
    let prefix = &stem[..cut];
    if prefix.is_empty() || prefix.contains('\n') {
        return None;
    }
    Some((
        prefix,
        five_digits(&suffix[1..6])?,
        five_digits(&suffix[10..15])?,
    ))
}

fn tag_of(prefix: &str) -> String {
    let start = prefix
        .char_indices()
        .rev()
        .find(|&(_, c)| !(c.is_ascii_alphanumeric() || c == '_'))
        .map_or(0, |(i, c)| i + c.len_utf8());
    match prefix[..start].chars().next_back() {
        Some('-' | '.') if start < prefix.len() => prefix[start..].to_ascii_uppercase(),
        _ => String::new(),
    }
}

pub fn split_info(path: &str) -> SplitInfo {
    let Some(stem) = path.strip_suffix(".gguf") else {
        return SplitInfo::default();
    };
    let (prefix, index, count) = match split_suffix(stem) {
        Some((prefix, index, count)) => (prefix.to_string(), index, count),
        None => (stem.to_string(), 1, 1),
    };
    let tag = tag_of(&prefix);
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
