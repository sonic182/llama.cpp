use std::ffi::OsStr;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::os::unix::ffi::OsStrExt;
use std::path::Path;

use llama::sys;

const ALIGNMENT: usize = sys::GGUF_DEFAULT_ALIGNMENT as usize;

pub fn pad(n: usize) -> usize {
    (n + ALIGNMENT - 1) & !(ALIGNMENT - 1)
}

pub fn path_of(bytes: &[u8]) -> &Path {
    Path::new(OsStr::from_bytes(bytes))
}

pub fn shown(bytes: &[u8]) -> std::borrow::Cow<'_, str> {
    String::from_utf8_lossy(bytes)
}

pub fn write_zeros(out: &mut impl Write, mut n: usize) -> io::Result<()> {
    const ZEROS: [u8; 4096] = [0; 4096];
    while n > 0 {
        let chunk = n.min(ZEROS.len());
        out.write_all(&ZEROS[..chunk])?;
        n -= chunk;
    }
    Ok(())
}

pub fn copy_range(
    input: &mut File,
    output: &mut File,
    offset: usize,
    len: usize,
) -> io::Result<()> {
    input.seek(SeekFrom::Start(offset as u64))?;
    let mut reader = Read::take(&*input, len as u64);
    let copied = io::copy(&mut reader, output)?;
    if copied != len as u64 {
        return Err(io::ErrorKind::UnexpectedEof.into());
    }
    Ok(())
}
