use std::fmt;

#[derive(Debug, PartialEq, Eq)]
pub struct InvalidRepo;

impl fmt::Display for InvalidRepo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("invalid HF repo format, expected <user>/<model>[:quant]")
    }
}

impl std::error::Error for InvalidRepo {}

pub fn split_repo_tag(spec: &str) -> Result<(&str, &str), InvalidRepo> {
    let parts: Vec<&str> = spec.split(':').collect();
    let repo = parts[0];
    let tag = if parts.len() > 1 {
        parts[parts.len() - 1]
    } else {
        ""
    };
    if repo.split('/').count() != 2 {
        return Err(InvalidRepo);
    }
    Ok((repo, tag))
}
