use llama_download::gguf::{self, SplitInfo};
use regex::Regex;

fn regex_split_info(path: &str) -> SplitInfo {
    let split = Regex::new(r"^(.+)-([0-9]{5})-(?i:of)-([0-9]{5})$").unwrap();
    let tag = Regex::new(r"[-.]([A-Za-z0-9_]+)$").unwrap();
    let Some(stem) = path.strip_suffix(".gguf") else {
        return SplitInfo::default();
    };
    let (prefix, index, count) = match split.captures(stem) {
        Some(c) => (
            c[1].to_string(),
            c[2].parse().unwrap(),
            c[3].parse().unwrap(),
        ),
        None => (stem.to_string(), 1, 1),
    };
    let tag = tag
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

#[test]
fn split_info_matches_the_original_regexes() {
    let cases = [
        "Model-Q4_K_M-00001-OF-00003.gguf",
        "Model-Q4_K_M-00001-Of-00003.gguf",
        "-00001-of-00002.gguf",
        "x-00001-of-00002.gguf",
        "Model-0001-of-00002.gguf",
        "Model-000001-of-00002.gguf",
        "Model-00001-of-000002.gguf",
        "Model-00001_of-00002.gguf",
        "Model-0000a-of-00002.gguf",
        "Mödel-Q4-00001-of-00002.gguf",
        "Model-Q4é-00001-of-00002.gguf",
        "Model-é.gguf",
        "a\nb-00001-of-00002.gguf",
        "Model-.gguf",
        "Model.-Q8_0.gguf",
        "Model Q8_0.gguf",
        "-Q8.gguf",
        "Q8.gguf",
        ".gguf",
        "é-00001-of-00002.gguf",
        "dir.v2/Model-Q4_0.gguf",
        "dir-v2/Model.gguf",
    ];
    for case in cases {
        assert_eq!(gguf::split_info(case), regex_split_info(case), "{case:?}");
    }
}
