use crate::gguf;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Wanted {
    pub mmproj: bool,
    pub mtp: bool,
    pub eagle3: bool,
    pub dflash: bool,
    pub dspark: bool,
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct Selection {
    pub preset: Option<usize>,
    pub primary: Option<usize>,
    pub model_files: Vec<usize>,
    pub mmproj: Option<usize>,
    pub mtp: Option<usize>,
    pub eagle3: Option<usize>,
    pub dflash: Option<usize>,
    pub dspark: Option<usize>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum SelectError {
    HfFileNotFound,
    NoGgufFiles,
}

struct Candidate {
    index: usize,
    depth: usize,
    exact: bool,
    diff: i64,
}

impl Candidate {
    fn beats(&self, best: &Candidate) -> bool {
        self.depth > best.depth
            || (self.depth == best.depth && self.exact && !best.exact)
            || (self.depth == best.depth && self.exact == best.exact && self.diff < best.diff)
    }
}

fn dir_components(path: &str) -> Vec<&str> {
    let mut parts: Vec<&str> = path.split('/').collect();
    parts.pop();
    parts
}

pub fn find_best_sibling<S: AsRef<str>>(
    paths: &[S],
    model: &str,
    keyword: &str,
    tag: &str,
) -> Option<usize> {
    let tag = tag.to_ascii_uppercase();
    let model_bits = if tag.is_empty() {
        gguf::quant_bits(model)
    } else {
        gguf::bits_of_tag(&tag)
    };
    let model_dirs = dir_components(model);
    let exact_marker = format!("-{tag}.");

    let mut best: Option<Candidate> = None;
    for (index, path) in paths.iter().enumerate() {
        let path = path.as_ref();
        if !path.ends_with(".gguf") || !path.contains(keyword) {
            continue;
        }
        let dirs = dir_components(path);
        if !model_dirs.starts_with(&dirs) {
            continue;
        }
        let candidate = Candidate {
            index,
            depth: dirs.len(),
            exact: !tag.is_empty() && path.to_ascii_uppercase().contains(&exact_marker),
            diff: (gguf::quant_bits(path) - model_bits).abs(),
        };
        if best.as_ref().is_none_or(|b| candidate.beats(b)) {
            best = Some(candidate);
        }
    }
    best.map(|b| b.index)
}

fn has_tag_token(path: &str, tag: &str) -> bool {
    let upper = path.to_ascii_uppercase();
    upper.contains(&format!("{tag}.")) || upper.contains(&format!("{tag}-"))
}

fn is_first_part(path: &str) -> bool {
    let split = gguf::split_info(path);
    split.count <= 1 || split.index == 1
}

pub fn find_best_model<S: AsRef<str>>(paths: &[S], tag: &str) -> Option<usize> {
    let tags: Vec<&str> = if tag.is_empty() {
        vec!["Q4_K_M", "Q8_0"]
    } else {
        vec![tag]
    };
    for tag in tags {
        let tag = tag.to_ascii_uppercase();
        let hit = paths.iter().position(|path| {
            let path = path.as_ref();
            gguf::is_model(path) && has_tag_token(path, &tag) && is_first_part(path)
        });
        if hit.is_some() {
            return hit;
        }
    }
    if tag.is_empty() {
        return paths
            .iter()
            .position(|path| gguf::is_model(path.as_ref()) && is_first_part(path.as_ref()));
    }
    None
}

pub fn select_hf_plan<S: AsRef<str>>(
    paths: &[S],
    hf_file: &str,
    tag: &str,
    wanted: Wanted,
) -> Result<Selection, SelectError> {
    if let Some(preset) = paths.iter().position(|path| path.as_ref() == "preset.ini") {
        return Ok(Selection {
            preset: Some(preset),
            ..Selection::default()
        });
    }

    let primary = if hf_file.is_empty() {
        find_best_model(paths, tag)
    } else {
        let found = paths.iter().position(|path| path.as_ref() == hf_file);
        Some(found.ok_or(SelectError::HfFileNotFound)?)
    };
    let model = primary.map_or("", |i| paths[i].as_ref());

    let mut selection = Selection {
        primary,
        ..Selection::default()
    };
    if let Some(i) = primary {
        selection.model_files = gguf::split_files(paths, i);
        if wanted.mmproj {
            selection.mmproj = find_best_sibling(paths, model, "mmproj", "");
        }
    }
    if wanted.mtp {
        selection.mtp = find_best_sibling(paths, model, "mtp-", tag);
    }
    if wanted.eagle3 {
        selection.eagle3 = find_best_sibling(paths, model, "eagle3-", tag);
    }
    if wanted.dflash {
        selection.dflash = find_best_sibling(paths, model, "dflash-", tag);
    }
    if wanted.dspark {
        selection.dspark = find_best_sibling(paths, model, "dspark-", tag);
    }

    let sidecar_found = [
        selection.mtp,
        selection.eagle3,
        selection.dflash,
        selection.dspark,
    ]
    .iter()
    .any(Option::is_some);
    if selection.primary.is_none() && !sidecar_found {
        return Err(SelectError::NoGgufFiles);
    }
    Ok(selection)
}
