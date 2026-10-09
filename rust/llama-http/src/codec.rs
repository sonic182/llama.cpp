pub type Pair = (Vec<u8>, Vec<u8>);

fn hex(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

pub fn decode(input: &[u8], plus_as_space: bool) -> Vec<u8> {
    let mut out = Vec::with_capacity(input.len());
    let mut i = 0;
    while i < input.len() {
        match input[i] {
            b'%' if i + 2 < input.len() => match (hex(input[i + 1]), hex(input[i + 2])) {
                (Some(hi), Some(lo)) => {
                    out.push(hi << 4 | lo);
                    i += 3;
                    continue;
                }
                _ => out.push(b'%'),
            },
            b'+' if plus_as_space => out.push(b' '),
            byte => out.push(byte),
        }
        i += 1;
    }
    out
}

pub fn encode_query_component(input: &[u8]) -> String {
    let mut out = String::with_capacity(input.len() * 3);
    for &byte in input {
        match byte {
            b'0'..=b'9' | b'a'..=b'z' | b'A'..=b'Z' | b'-' | b'.' | b'_' | b'~' => {
                out.push(char::from(byte));
            }
            b' ' => out.push('+'),
            _ => out.push_str(&format!("%{byte:02X}")),
        }
    }
    out
}

pub fn parse_query(query: &str) -> Vec<Pair> {
    let mut seen = std::collections::HashSet::new();
    let mut params: Vec<Pair> = Vec::new();
    for span in query.split('&') {
        if !seen.insert(span) {
            continue;
        }
        let (key, value) = span.split_once('=').unwrap_or((span, ""));
        if !key.is_empty() {
            params.push((decode(key.as_bytes(), true), decode(value.as_bytes(), true)));
        }
    }
    params.sort_by(|a, b| a.0.cmp(&b.0));
    params
}

pub fn build_query_string(params: &[Pair]) -> String {
    params
        .iter()
        .map(|(key, value)| {
            format!(
                "{}={}",
                encode_query_component(key),
                encode_query_component(value)
            )
        })
        .collect::<Vec<_>>()
        .join("&")
}

pub fn origin_is_localhost(origin: &str) -> bool {
    parse_origin_host(origin)
        .is_some_and(|host| matches!(host.as_str(), "localhost" | "127.0.0.1" | "::1"))
}

fn parse_origin_host(url: &str) -> Option<String> {
    let (scheme, rest) = url.split_once("://")?;
    if scheme != "http" && scheme != "https" {
        return None;
    }
    let rest = rest.split_once('@').map_or(rest, |(_, after)| after);
    let authority = rest.find('/').map_or(rest, |slash| &rest[..slash]);

    let (host, port) = if let Some(inner) = authority.strip_prefix('[') {
        let close = inner.find(']')?;
        (&inner[..close], inner[close + 1..].strip_prefix(':'))
    } else {
        match authority.split_once(':') {
            Some((host, port)) => (host, Some(port)),
            None => (authority, None),
        }
    };
    if let Some(port) = port.filter(|p| !p.is_empty()) {
        parse_leading_int(port)?;
    }
    Some(host.to_owned())
}

fn parse_leading_int(text: &str) -> Option<i32> {
    let trimmed = text.trim_start();
    let (negative, digits) = match trimmed.as_bytes().first()? {
        b'-' => (true, &trimmed[1..]),
        b'+' => (false, &trimmed[1..]),
        _ => (false, trimmed),
    };
    let end = digits
        .bytes()
        .position(|b| !b.is_ascii_digit())
        .unwrap_or(digits.len());
    if end == 0 {
        return None;
    }
    let magnitude: i64 = digits[..end].parse().ok()?;
    i32::try_from(if negative { -magnitude } else { magnitude }).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_like_httplib() {
        assert_eq!(decode(b"a%3A%3Ab", false), b"a::b");
        assert_eq!(decode(b"a+b%2", true), b"a b%2");
        assert_eq!(decode(b"a+b", false), b"a+b");
        assert_eq!(decode(b"%zz%41", false), b"%zzA");
    }

    #[test]
    fn query_is_deduplicated_sorted_and_reencoded() {
        let params = parse_query("b=2&a=1&a=1&=x&flag&c=%20+");
        let text: Vec<_> = params
            .iter()
            .map(|(k, v)| (String::from_utf8_lossy(k), String::from_utf8_lossy(v)))
            .collect();
        assert_eq!(
            text,
            [
                ("a".into(), "1".into()),
                ("b".into(), "2".into()),
                ("c".into(), "  ".into()),
                ("flag".into(), "".into()),
            ]
        );
        assert_eq!(build_query_string(&params), "a=1&b=2&c=++&flag=");
    }

    #[test]
    fn localhost_origins() {
        assert!(origin_is_localhost("http://localhost:3000"));
        assert!(origin_is_localhost("https://127.0.0.1"));
        assert!(origin_is_localhost("http://[::1]:8080"));
        assert!(!origin_is_localhost("http://evil.example"));
        assert!(!origin_is_localhost("http://localhost:abc"));
        assert!(!origin_is_localhost("ftp://localhost"));
        assert!(!origin_is_localhost("localhost"));
    }
}
