pub fn process_escapes(input: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(input.len());
    let mut i = 0;
    while i < input.len() {
        if input[i] == b'\\' && i + 1 < input.len() {
            i += 1;
            match input[i] {
                b'n' => out.push(b'\n'),
                b'r' => out.push(b'\r'),
                b't' => out.push(b'\t'),
                b'\'' => out.push(b'\''),
                b'"' => out.push(b'"'),
                b'\\' => out.push(b'\\'),
                b'x' if i + 2 < input.len() => match hex_pair(input[i + 1], input[i + 2]) {
                    Some(value) => {
                        out.push(value as u8);
                        i += 2;
                    }
                    None => out.extend_from_slice(b"\\x"),
                },
                other => out.extend_from_slice(&[b'\\', other]),
            }
        } else {
            out.push(input[i]);
        }
        i += 1;
    }
    out
}

fn hex_pair(a: u8, b: u8) -> Option<i64> {
    let digit = |c: u8| (c as char).to_digit(16).map(i64::from);
    match a {
        b'+' => digit(b),
        b'-' => digit(b).map(|d| -d),
        c if c.is_ascii_whitespace() || c == 0x0b => digit(b),
        _ => Some(digit(a)? * 16 + digit(b)?),
    }
}

#[cfg(test)]
mod tests {
    use super::process_escapes;

    #[test]
    fn matches_cpp_string_process_escapes() {
        let cases: &[(&[u8], &[u8])] = &[
            (b"a\\nb\\t\\\\", b"a\nb\t\\"),
            (b"\\x41B", b"AB"),
            (b"\\x41", b"A"),
            (b"\\x4", b"\\x4"),
            (b"\\xZZ", b"\\xZZ"),
            (b"\\x-1", &[0xff]),
            (b"\\q\\", b"\\q\\"),
        ];
        for (input, expected) in cases {
            assert_eq!(process_escapes(input), *expected, "input {input:?}");
        }
    }
}
