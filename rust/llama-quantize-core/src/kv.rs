use std::io::Write;

use crate::cnum::{atof, atol};

pub const KEY_MAX: usize = 128;
pub const STR_MAX: usize = 127;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(Vec<u8>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Override {
    pub key: Vec<u8>,
    pub value: Value,
}

impl Override {
    pub fn str(key: &str, value: &[u8]) -> Self {
        Override {
            key: key.as_bytes().to_vec(),
            value: Value::Str(value[..value.len().min(STR_MAX)].to_vec()),
        }
    }

    pub fn int(key: &str, value: i64) -> Self {
        Override {
            key: key.as_bytes().to_vec(),
            value: Value::Int(value),
        }
    }
}

fn com_err(err: &mut dyn Write, msg: &str, data: &[u8], suffix: &str) {
    let _ = write!(err, "cmn  string_parse: string_parse_kv_override: {msg} '");
    let _ = err.write_all(data);
    let _ = writeln!(err, "'{suffix}");
}

pub fn parse(data: &[u8], err: &mut dyn Write) -> Option<Override> {
    let Some(sep) = data
        .iter()
        .position(|&b| b == b'=')
        .filter(|&p| p < KEY_MAX)
    else {
        com_err(err, "malformed KV override", data, "");
        return None;
    };
    let key = data[..sep].to_vec();
    let rest = &data[sep + 1..];
    let value = if let Some(v) = rest.strip_prefix(b"int:") {
        Value::Int(atol(v))
    } else if let Some(v) = rest.strip_prefix(b"float:") {
        Value::Float(atof(v))
    } else if let Some(v) = rest.strip_prefix(b"bool:") {
        match v {
            b"true" => Value::Bool(true),
            b"false" => Value::Bool(false),
            _ => {
                com_err(err, "invalid boolean value for KV override", data, "");
                return None;
            }
        }
    } else if let Some(v) = rest.strip_prefix(b"str:") {
        if v.len() > STR_MAX {
            com_err(
                err,
                "malformed KV override",
                data,
                ", value cannot exceed 127 chars",
            );
            return None;
        }
        Value::Str(v.to_vec())
    } else {
        com_err(err, "invalid type for KV override", data, "");
        return None;
    };
    Some(Override { key, value })
}
