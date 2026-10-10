use std::collections::BTreeMap;
use std::io::Write;

use crate::cnum::{c_str, f32_to_c_int};

#[derive(Debug, Default, Clone, PartialEq)]
pub struct Entry {
    pub sums: Vec<f32>,
    pub counts: Vec<i64>,
}

impl Entry {
    pub fn from_gguf(sums: Vec<f32>, counts: &[f32]) -> Option<Entry> {
        if counts.is_empty() {
            return None;
        }
        Some(Entry {
            sums,
            counts: counts.iter().map(|&c| lround(c)).collect(),
        })
    }
}

fn lround(c: f32) -> i64 {
    if c.is_finite() && c.abs() < 9_223_372_036_854_775_808.0 {
        c.round() as i64
    } else {
        i64::MIN
    }
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct Imatrix {
    pub entries: BTreeMap<Vec<u8>, Entry>,
    pub datasets: Vec<Vec<u8>>,
    pub chunk_count: i32,
    pub chunk_size: i32,
    pub is_legacy: bool,
    pub has_metadata: bool,
}

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
    failed: bool,
}

impl<'a> Reader<'a> {
    fn bytes(&mut self, n: usize) -> &'a [u8] {
        if self.failed {
            return &[];
        }
        let available = self.data.len() - self.pos;
        let take = n.min(available);
        let out = &self.data[self.pos..self.pos + take];
        self.pos += take;
        if take < n {
            self.failed = true;
        }
        out
    }

    fn i32(&mut self) -> i32 {
        let mut raw = [0u8; 4];
        let got = self.bytes(4);
        raw[..got.len()].copy_from_slice(got);
        i32::from_le_bytes(raw)
    }

    fn at_end(&self) -> bool {
        self.pos >= self.data.len()
    }
}

pub fn parse_legacy(data: &[u8], fname: &[u8]) -> Result<Imatrix, Vec<u8>> {
    const FUNC: &str = "common_imatrix_load_legacy";
    let mut imatrix = Imatrix::default();
    let mut r = Reader {
        data,
        pos: 0,
        failed: false,
    };

    let n_entries = r.i32();
    if r.failed || n_entries < 1 {
        return Err(message(format_args!("{FUNC}: no data in file "), fname));
    }

    for i in 0..n_entries {
        let len = r.i32();
        let name = match usize::try_from(len) {
            Ok(len) => c_str(r.bytes(len)).to_vec(),
            Err(_) => {
                r.failed = true;
                Vec::new()
            }
        };
        if r.failed {
            return Err(message(
                format_args!("{FUNC}: failed reading name for entry {} from ", i + 1),
                fname,
            ));
        }

        let ncall = r.i32();
        let nval = r.i32();
        if r.failed || nval < 1 {
            return Err(format!("{FUNC}: failed reading number of values for entry {i}\n").into());
        }

        let raw = r.bytes(nval as usize * 4);
        if r.failed {
            return Err(format!("{FUNC}: failed reading data for entry {i}\n").into());
        }
        let sums = raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        imatrix.entries.insert(
            name,
            Entry {
                sums,
                counts: vec![i64::from(ncall)],
            },
        );
    }

    if !r.at_end() {
        imatrix.chunk_count = r.i32();
        if !r.failed {
            let len = r.i32();
            if !r.failed && len > 0 {
                let dataset = c_str(r.bytes(len as usize)).to_vec();
                if !r.failed {
                    imatrix.datasets.push(dataset);
                }
            }
        }
    }

    imatrix.chunk_size = 0;
    imatrix.is_legacy = true;
    Ok(imatrix)
}

fn message(prefix: std::fmt::Arguments<'_>, fname: &[u8]) -> Vec<u8> {
    let mut out = prefix.to_string().into_bytes();
    out.extend_from_slice(fname);
    out.push(b'\n');
    out
}

pub type Data = BTreeMap<Vec<u8>, Vec<f32>>;

pub fn normalize(loaded: &Imatrix, trace: bool, out: &mut dyn Write) -> Data {
    let mut data = Data::new();
    for (name, entry) in &loaded.entries {
        let mut e = vec![0f32; entry.sums.len()];
        if !loaded.is_legacy {
            let ncounts = entry.counts.len();
            let ne0 = entry.sums.len() / ncounts;
            for (j, &count) in entry.counts.iter().enumerate() {
                let count = count as f32;
                let row = j * ne0..(j + 1) * ne0;
                if count > 0.0 {
                    for (dst, src) in e[row.clone()].iter_mut().zip(&entry.sums[row]) {
                        *dst = src / count;
                    }
                } else {
                    e[row].fill(1.0);
                }
            }
            if trace {
                let max_count = entry
                    .counts
                    .iter()
                    .map(|&c| c as f32)
                    .fold(0.0f32, |m, c| if c > m { c } else { m });
                let _ = write!(
                    out,
                    "load_imatrix: loaded data (size = {:6}, n_tokens = {:6}, n_chunks = {:6}) for '",
                    e.len() as i32,
                    f32_to_c_int(max_count),
                    f32_to_c_int(max_count / loaded.chunk_size as f32),
                );
                let _ = out.write_all(name);
                let _ = out.write_all(b"'\n");
            }
        } else {
            let ncall = entry.counts.first().copied().unwrap_or(0);
            if ncall > 0 {
                for (dst, src) in e.iter_mut().zip(&entry.sums) {
                    *dst = src / ncall as f32;
                }
            } else {
                e.copy_from_slice(&entry.sums);
            }
            if trace {
                let _ = write!(
                    out,
                    "load_imatrix: loaded data (size = {:6}, ncall = {:6}) for '",
                    e.len() as i32,
                    ncall as i32,
                );
                let _ = out.write_all(name);
                let _ = out.write_all(b"'\n");
            }
        }
        data.insert(name.clone(), e);
    }
    data
}

pub fn filter(data: &mut Data, included: &[Vec<u8>], excluded: &[Vec<u8>]) {
    for pattern in excluded {
        data.retain(|name, _| !contains(name, pattern));
    }
    if !included.is_empty() {
        data.retain(|name, _| included.iter().any(|p| contains(name, p)));
    }
}

fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    needle.is_empty() || haystack.windows(needle.len()).any(|w| w == needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn legacy(entries: &[(&[u8], i32, &[f32])], tail: &[u8]) -> Vec<u8> {
        let mut buf = (entries.len() as i32).to_le_bytes().to_vec();
        for (name, ncall, sums) in entries {
            buf.extend((name.len() as i32).to_le_bytes());
            buf.extend(*name);
            buf.extend(ncall.to_le_bytes());
            buf.extend((sums.len() as i32).to_le_bytes());
            sums.iter().for_each(|s| buf.extend(s.to_le_bytes()));
        }
        buf.extend(tail);
        buf
    }

    #[test]
    fn legacy_reader_matches_cpp_edge_cases() {
        let mut tail = 7i32.to_le_bytes().to_vec();
        tail.extend(4i32.to_le_bytes());
        tail.extend(b"d\0ta");
        let m = parse_legacy(&legacy(&[(b"a\0b", 2, &[4.0, 6.0])], &tail), b"f").unwrap();
        assert_eq!(m.entries[b"a".as_slice()].counts, vec![2]);
        assert_eq!(m.datasets, vec![b"d".to_vec()]);
        assert_eq!(m.chunk_count, 7);
        assert!(m.is_legacy);

        let m = parse_legacy(&legacy(&[(b"x", 1, &[1.0])], &[9, 0]), b"f").unwrap();
        assert_eq!(m.chunk_count, 9);
        assert!(m.datasets.is_empty());

        let full = legacy(&[(b"x", 1, &[1.0, 2.0])], &[]);
        assert_eq!(
            parse_legacy(&full[..full.len() - 1], b"f").unwrap_err(),
            b"common_imatrix_load_legacy: failed reading data for entry 0\n"
        );
        assert_eq!(
            parse_legacy(&[], b"in.dat").unwrap_err(),
            b"common_imatrix_load_legacy: no data in file in.dat\n"
        );
        let mut huge = 1i32.to_le_bytes().to_vec();
        huge.extend(i32::MAX.to_le_bytes());
        assert_eq!(
            parse_legacy(&huge, b"in.dat").unwrap_err(),
            b"common_imatrix_load_legacy: failed reading name for entry 1 from in.dat\n"
        );
    }

    #[test]
    fn gguf_entry_rejects_empty_counts_and_rounds_like_lround() {
        assert_eq!(Entry::from_gguf(vec![1.0], &[]), None);
        let e = Entry::from_gguf(vec![], &[2.5, -2.5, f32::INFINITY, f32::NAN, 1e19]).unwrap();
        assert_eq!(e.counts, vec![3, -3, i64::MIN, i64::MIN, i64::MIN]);
    }

    #[test]
    fn normalize_divides_by_counts_and_filters_by_substring() {
        let mut m = Imatrix::default();
        m.entries.insert(
            b"blk.0.attn_q".to_vec(),
            Entry {
                sums: vec![2.0, 4.0, 3.0, 9.0],
                counts: vec![2, 0],
            },
        );
        m.entries.insert(
            b"blk.0.ffn_up".to_vec(),
            Entry {
                sums: vec![1.0],
                counts: vec![4],
            },
        );
        let mut trace = Vec::new();
        let mut data = normalize(&m, true, &mut trace);
        assert_eq!(data[b"blk.0.attn_q".as_slice()], vec![1.0, 2.0, 1.0, 1.0]);
        assert_eq!(
            String::from_utf8(trace).unwrap().lines().next().unwrap(),
            "load_imatrix: loaded data (size =      4, n_tokens =      2, n_chunks = -2147483648) for 'blk.0.attn_q'"
        );
        filter(
            &mut data,
            &[b"attn".to_vec(), b"blk".to_vec()],
            &[b"ffn".to_vec()],
        );
        assert_eq!(data.keys().collect::<Vec<_>>(), vec![b"blk.0.attn_q"]);
    }
}
