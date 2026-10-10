use serde_json::Value;

use crate::grisu::grisu2;

const MAX_EXP: i32 = 15;
const MIN_EXP: i32 = -4;

pub fn dump(value: &Value) -> String {
    let mut out = String::new();
    write_value(value, &mut out);
    out
}

pub fn dump_string(s: &str) -> String {
    let mut out = String::new();
    write_string(s, &mut out);
    out
}

fn write_value(value: &Value, out: &mut String) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                out.push_str(&i.to_string());
            } else if let Some(u) = n.as_u64() {
                out.push_str(&u.to_string());
            } else {
                write_float(n.as_f64().unwrap_or(0.0), out);
            }
        }
        Value::String(s) => write_string(s, out),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_value(item, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            for (i, (key, item)) in map.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_string(key, out);
                out.push(':');
                write_value(item, out);
            }
            out.push('}');
        }
    }
}

fn write_string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

fn write_float(x: f64, out: &mut String) {
    if !x.is_finite() {
        out.push_str("null");
        return;
    }
    let mut x = x;
    if x.is_sign_negative() {
        out.push('-');
        x = -x;
    }
    if x == 0.0 {
        out.push_str("0.0");
        return;
    }
    let (digits, decimal_exponent) = grisu2(x);
    let digits = String::from_utf8_lossy(&digits).into_owned();
    let k = digits.len() as i32;
    let n = k + decimal_exponent;

    if k <= n && n <= MAX_EXP {
        out.push_str(&digits);
        out.push_str(&"0".repeat((n - k) as usize));
        out.push_str(".0");
    } else if 0 < n && n <= MAX_EXP {
        out.push_str(&digits[..n as usize]);
        out.push('.');
        out.push_str(&digits[n as usize..]);
    } else if MIN_EXP < n && n <= 0 {
        out.push_str("0.");
        out.push_str(&"0".repeat((-n) as usize));
        out.push_str(&digits);
    } else {
        out.push_str(&digits[..1]);
        if k > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        out.push('e');
        let e = n - 1;
        out.push(if e < 0 { '-' } else { '+' });
        let abs = e.unsigned_abs();
        if abs < 10 {
            out.push('0');
        }
        out.push_str(&abs.to_string());
    }
}
