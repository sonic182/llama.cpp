use std::ffi::{CString, c_char};

pub fn c_str(bytes: &[u8]) -> &[u8] {
    match bytes.iter().position(|&b| b == 0) {
        Some(end) => &bytes[..end],
        None => bytes,
    }
}

fn cstring(bytes: &[u8]) -> CString {
    CString::new(c_str(bytes)).expect("c_str stops at the first NUL")
}

pub fn atoi(s: &[u8]) -> i32 {
    let s = cstring(s);
    unsafe { libc::atoi(s.as_ptr()) }
}

pub fn atol(s: &[u8]) -> i64 {
    let s = cstring(s);
    unsafe { libc::atol(s.as_ptr()) as i64 }
}

pub fn atof(s: &[u8]) -> f64 {
    let s = cstring(s);
    unsafe { libc::atof(s.as_ptr()) }
}

pub fn stoi(s: &[u8]) -> Option<i32> {
    let s = cstring(s);
    let mut end: *mut c_char = std::ptr::null_mut();
    let value = unsafe { libc::strtol(s.as_ptr(), &mut end, 10) };
    if end.cast_const() == s.as_ptr() {
        return None;
    }
    i32::try_from(value).ok()
}

pub fn is_c_space(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r')
}

pub fn strerror(errno: i32) -> Vec<u8> {
    let msg = unsafe { std::ffi::CStr::from_ptr(libc::strerror(errno)) };
    msg.to_bytes().to_vec()
}

pub fn f32_to_c_int(v: f32) -> i32 {
    if v.is_nan() || !(-2_147_483_648.0..2_147_483_648.0).contains(&v) {
        i32::MIN
    } else {
        v as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn number_parsing_follows_libc() {
        assert_eq!(stoi(b" 7x"), Some(7));
        assert_eq!(stoi(b"-3"), Some(-3));
        assert_eq!(stoi(b"x"), None);
        assert_eq!(stoi(b""), None);
        assert_eq!(stoi(b"99999999999"), None);
        assert_eq!(atoi(b"abc"), 0);
        assert_eq!(atol(b"42x"), 42);
        assert_eq!(atof(b"1.5e1"), 15.0);
        assert_eq!(f32_to_c_int(f32::INFINITY), i32::MIN);
        assert_eq!(f32_to_c_int(3.9), 3);
    }
}
