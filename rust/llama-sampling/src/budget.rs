use llama::sys::{llama_token, llama_token_data};

use crate::aho_corasick::AhoCorasick;
use crate::log::trace;

#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum State {
    Idle = 0,
    Counting = 1,
    Forcing = 2,
    WaitingUtf8 = 3,
    Done = 4,
}

impl State {
    pub fn from_raw(raw: i32) -> Option<Self> {
        Some(match raw {
            0 => Self::Idle,
            1 => Self::Counting,
            2 => Self::Forcing,
            3 => Self::WaitingUtf8,
            4 => Self::Done,
            _ => return None,
        })
    }
}

#[derive(Clone)]
struct Matcher {
    seqs: Vec<Vec<llama_token>>,
    ac: AhoCorasick,
    state: usize,
}

impl Matcher {
    fn new(seqs: &[Vec<llama_token>]) -> Self {
        let mut unique: Vec<Vec<llama_token>> = Vec::new();
        for seq in seqs {
            if !seq.is_empty() && !unique.contains(seq) {
                unique.push(seq.clone());
            }
        }
        let symbols: Vec<Vec<u32>> = unique
            .iter()
            .map(|seq| seq.iter().map(|&t| t as u32).collect())
            .collect();
        Self {
            ac: AhoCorasick::new(symbols.iter().map(Vec::as_slice)),
            seqs: unique,
            state: 0,
        }
    }

    fn advance(&mut self, token: llama_token) -> Option<usize> {
        self.state = self.ac.next(self.state, token as u32);
        let matched = self.ac.match_pattern(self.state);
        if matched.is_some() {
            self.state = 0;
        }
        matched
    }

    fn reset(&mut self) {
        self.state = 0;
    }
}

const ACCEPT: &std::ffi::CStr = c"common_reasoning_budget_accept";
const FORCE: &std::ffi::CStr = c"common_reasoning_budget_force";

#[derive(Clone)]
pub struct Budget {
    start: Matcher,
    end: Matcher,
    forced: Vec<llama_token>,
    budget: i32,
    remaining: i32,
    state: State,
    force_pos: usize,
    end_match: Option<usize>,
}

impl Budget {
    pub fn new(
        start_seqs: &[Vec<llama_token>],
        end_seqs: &[Vec<llama_token>],
        forced: Vec<llama_token>,
        budget: i32,
        mut state: State,
    ) -> Self {
        if state == State::Counting && budget <= 0 {
            state = State::Forcing;
        }
        Self {
            start: Matcher::new(start_seqs),
            end: Matcher::new(end_seqs),
            forced,
            budget,
            remaining: budget,
            state,
            force_pos: 0,
            end_match: None,
        }
    }

    pub fn state(&self) -> State {
        self.state
    }

    pub fn end_match(&self) -> Option<&[llama_token]> {
        self.end_match.map(|i| self.end.seqs[i].as_slice())
    }

    pub fn accept(&mut self, token: llama_token, utf8_complete: impl FnOnce() -> bool) {
        match self.state {
            State::Idle => {
                if self.start.advance(token).is_some() {
                    self.activate("activated");
                }
            }
            State::Counting | State::WaitingUtf8 => {
                if let Some(matched) = self.end.advance(token) {
                    self.state = State::Done;
                    self.end_match = Some(matched);
                    trace(ACCEPT, "deactivated (natural end)\n");
                    return;
                }
                let complete = utf8_complete();
                if self.state == State::WaitingUtf8 {
                    if complete {
                        self.start_forcing();
                        trace(ACCEPT, "UTF-8 complete, now forcing end sequence\n");
                    }
                } else {
                    self.remaining -= 1;
                    if self.remaining <= 0 {
                        if complete {
                            self.start_forcing();
                            trace(ACCEPT, "budget exhausted, forcing end sequence\n");
                        } else {
                            self.state = State::WaitingUtf8;
                            self.end.reset();
                            trace(ACCEPT, "budget exhausted, waiting for UTF-8 completion\n");
                        }
                    }
                }
            }
            State::Forcing => {
                let matched = self.end.advance(token);
                self.force_pos += 1;
                if self.force_pos >= self.forced.len() {
                    self.state = State::Done;
                    self.end_match = matched;
                    trace(ACCEPT, "forced sequence complete, done\n");
                }
            }
            State::Done => {
                if self.start.advance(token).is_some() {
                    self.end.reset();
                    self.end_match = None;
                    self.activate("re-activated on new start tag");
                }
            }
        }
    }

    fn activate(&mut self, what: &str) {
        self.state = State::Counting;
        self.remaining = self.budget;
        trace(ACCEPT, &format!("{what}, budget={} tokens\n", self.budget));
        if self.remaining <= 0 {
            self.state = State::Forcing;
            self.force_pos = 0;
            trace(ACCEPT, "budget=0, forcing immediately\n");
        }
    }

    fn start_forcing(&mut self) {
        self.state = State::Forcing;
        self.force_pos = 0;
        self.end.reset();
    }

    pub fn apply(&self, cur: &mut [llama_token_data]) {
        if self.state != State::Forcing {
            return;
        }
        let Some(&forced) = self.forced.get(self.force_pos) else {
            return;
        };
        for td in cur.iter_mut().filter(|td| td.id != forced) {
            td.logit = f32::NEG_INFINITY;
        }
    }

    pub fn reset(&mut self) {
        self.state = State::Idle;
        self.remaining = self.budget;
        self.start.reset();
        self.end.reset();
        self.force_pos = 0;
        self.end_match = None;
    }

    pub fn force(&mut self) -> bool {
        if self.state != State::Counting {
            return false;
        }
        self.start_forcing();
        trace(FORCE, "forced into forcing state (manual transition)\n");
        true
    }
}

pub fn utf8_is_complete(s: &[u8]) -> bool {
    if s.is_empty() {
        return true;
    }
    for i in 1..=s.len().min(4) {
        let c = s[s.len() - i];
        if c & 0xC0 != 0x80 {
            let expected = match c {
                0xF0.. => 4,
                0xE0.. => 3,
                0xC0.. => 2,
                _ => 1,
            };
            return i >= expected;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn forced_token(b: &Budget) -> Option<llama_token> {
        let mut cur: Vec<llama_token_data> = (0..8)
            .map(|id| llama_token_data {
                id,
                logit: 0.0,
                p: 0.0,
            })
            .collect();
        b.apply(&mut cur);
        let finite: Vec<_> = cur.iter().filter(|t| t.logit.is_finite()).collect();
        (finite.len() == 1).then(|| finite[0].id)
    }

    #[test]
    fn counts_waits_for_utf8_then_forces_and_rearms() {
        let mut b = Budget::new(
            &[vec![1]],
            &[vec![2], vec![3, 2]],
            vec![5, 2],
            2,
            State::Idle,
        );
        b.accept(1, || unreachable!());
        assert_eq!(b.state(), State::Counting);
        b.accept(4, || true);
        b.accept(4, || false);
        assert_eq!(b.state(), State::WaitingUtf8);
        assert_eq!(forced_token(&b), None);
        b.accept(4, || true);
        assert_eq!(forced_token(&b), Some(5));
        b.accept(5, || unreachable!());
        assert_eq!(forced_token(&b), Some(2));
        b.accept(2, || unreachable!());
        assert_eq!(b.state(), State::Done);
        assert_eq!(b.end_match(), Some([2].as_slice()));

        b.accept(1, || unreachable!());
        assert_eq!(b.state(), State::Counting);
        assert_eq!(b.end_match(), None);
        b.accept(3, || true);
        b.accept(2, || unreachable!());
        assert_eq!(b.end_match(), Some([3, 2].as_slice()));
    }

    #[test]
    fn utf8_completeness_looks_at_the_last_sequence() {
        assert!(utf8_is_complete(b""));
        assert!(utf8_is_complete("abc\u{e9}".as_bytes()));
        assert!(utf8_is_complete("\u{1F600}".as_bytes()));
        assert!(!utf8_is_complete(b"hello\xC3"));
        assert!(!utf8_is_complete(b"\xF0\x9F\x98"));
        assert!(!utf8_is_complete(b"\x80"));
    }
}
