use std::collections::BTreeMap;
use std::io::{self, IsTerminal, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard};

use crate::remote::{Callback, Progress};

const ELLIPSIS: &str = "\u{2026}";
const FULL: &str = "\u{2500}";
const HALF: &str = "\u{2574}";

pub struct Board<W> {
    state: Mutex<State<W>>,
    next_id: AtomicUsize,
}

struct State<W> {
    out: W,
    tty: bool,
    lines: BTreeMap<usize, usize>,
    max_line: usize,
}

impl<W> State<W> {
    fn cleanup(&mut self, id: usize) {
        self.lines.remove(&id);
        if self.lines.is_empty() {
            self.max_line = 0;
        }
    }
}

impl<W: Write> Board<W> {
    pub const fn new(out: W, tty: bool) -> Self {
        Board {
            state: Mutex::new(State {
                out,
                tty,
                lines: BTreeMap::new(),
                max_line: 0,
            }),
            next_id: AtomicUsize::new(0),
        }
    }

    pub fn bar(&self) -> Bar<'_, W> {
        Bar {
            board: self,
            id: self.next_id.fetch_add(1, Ordering::Relaxed),
            name: Mutex::new((String::new(), 0)),
        }
    }

    fn lock(&self) -> MutexGuard<'_, State<W>> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn into_inner(self) -> W {
        self.state
            .into_inner()
            .unwrap_or_else(|e| e.into_inner())
            .out
    }
}

pub struct Bar<'a, W> {
    board: &'a Board<W>,
    id: usize,
    name: Mutex<(String, usize)>,
}

fn display_name(url: &str) -> (String, usize) {
    let mut name = match url.rfind('/') {
        Some(pos) => &url[pos + 1..],
        None => url,
    };
    if let Some(pos) = name.find('?') {
        name = &name[..pos];
    }
    let mut len = 0;
    for (i, _) in name.char_indices() {
        if len == 39 {
            return (format!("{}{ELLIPSIS}", &name[..i]), 40);
        }
        len += 1;
    }
    (name.to_string(), len)
}

impl<W: Write + Send> Callback for Bar<'_, W> {
    fn on_start(&self, p: &Progress) {
        *self.name.lock().unwrap_or_else(|e| e.into_inner()) = display_name(&p.url);
    }

    fn on_update(&self, p: &Progress) {
        let mut state = self.board.lock();
        if p.total == 0 || !state.tty {
            return;
        }
        let (name, len) = self.name.lock().unwrap_or_else(|e| e.into_inner()).clone();

        if !state.lines.contains_key(&self.id) {
            let line = state.max_line;
            state.lines.insert(self.id, line);
            state.max_line += 1;
            let _ = state.out.write_all(b"\n");
        }
        let lines_up = state.max_line - state.lines[&self.id];

        let bar = (55 - len as u128) * 2;
        let pct = 100 * u128::from(p.downloaded) / u128::from(p.total);
        let pos = bar * u128::from(p.downloaded) / u128::from(p.total);

        let mut line = String::new();
        if lines_up > 0 {
            line += &format!("\x1b[{lines_up}A");
        }
        line += &format!("\rDownloading {name} ");
        for i in (0..bar).step_by(2) {
            line += if i + 1 < pos {
                FULL
            } else if i < pos {
                HALF
            } else {
                " "
            };
        }
        line += &format!("{pct:>4}%\x1b[K");
        if lines_up > 0 {
            line += &format!("\x1b[{lines_up}B");
        }
        line += "\r";
        let _ = state.out.write_all(line.as_bytes());
        let _ = state.out.flush();

        if p.downloaded == p.total {
            state.cleanup(self.id);
        }
    }

    fn on_done(&self, _p: &Progress, _ok: bool) {
        self.board.lock().cleanup(self.id);
    }
}

pub struct Stdout;

impl Write for Stdout {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        io::stdout().write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        io::stdout().flush()
    }
}

static STDOUT: Board<Stdout> = Board::new(Stdout, false);

pub fn stdout_bar() -> StdoutBar {
    StdoutBar(STDOUT.bar())
}

pub struct StdoutBar(Bar<'static, Stdout>);

impl Callback for StdoutBar {
    fn on_start(&self, p: &Progress) {
        self.0.on_start(p)
    }

    fn on_update(&self, p: &Progress) {
        STDOUT.lock().tty = io::stdout().is_terminal();
        self.0.on_update(p)
    }

    fn on_done(&self, p: &Progress, ok: bool) {
        self.0.on_done(p, ok)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn progress(url: &str, downloaded: u64, total: u64) -> Progress {
        Progress {
            url: url.to_string(),
            downloaded,
            total,
            cached: false,
        }
    }

    fn expected(ascii: &str) -> String {
        ascii
            .replace('#', FULL)
            .replace('>', HALF)
            .replace('~', ELLIPSIS)
            .replace("e!", "\u{e9}")
    }

    fn render(tty: bool, run: impl FnOnce(&Board<Vec<u8>>)) -> String {
        let board = Board::new(Vec::new(), tty);
        run(&board);
        String::from_utf8(board.into_inner()).unwrap()
    }

    fn single(board: &Board<Vec<u8>>) {
        let url = "https://huggingface.co/org/repo/resolve/main/model-Q4_K_M.gguf?download=true";
        let bar = board.bar();
        bar.on_start(&progress(url, 0, 0));
        bar.on_update(&progress(url, 0, 0));
        bar.on_update(&progress(url, 0, 1000));
        bar.on_update(&progress(url, 333, 1000));
        bar.on_update(&progress(url, 500, 1000));
        bar.on_update(&progress(url, 1000, 1000));
        bar.on_done(&progress(url, 1000, 1000), true);
    }

    #[test]
    fn single_bar_matches_cpp() {
        assert_eq!(
            render(true, single),
            expected(
                "\n\x1b[1A\rDownloading model-Q4_K_M.gguf                                          0%\x1b[K\x1b[1B\r\x1b[1A\rDownloading model-Q4_K_M.gguf ############>                           33%\x1b[K\x1b[1B\r\x1b[1A\rDownloading model-Q4_K_M.gguf ###################                     50%\x1b[K\x1b[1B\r\x1b[1A\rDownloading model-Q4_K_M.gguf ###################################### 100%\x1b[K\x1b[1B\r"
            )
        );
    }

    #[test]
    fn long_name_is_cut_on_a_char_boundary() {
        let out = render(true, |board| {
            let url = "https://example.com/files/\u{e9}tude-a-very-long-model-file-name-that-is-cut-Q8_0.gguf";
            let bar = board.bar();
            bar.on_start(&progress(url, 0, 0));
            bar.on_update(&progress(url, 7, 10));
            bar.on_done(&progress(url, 7, 10), false);
        });
        assert_eq!(
            out,
            expected(
                "\n\x1b[1A\rDownloading e!tude-a-very-long-model-file-name-that-~ ##########>      70%\x1b[K\x1b[1B\r"
            )
        );
    }

    #[test]
    fn concurrent_bars_keep_their_lines() {
        let out = render(true, |board| {
            let (ua, ub) = ("https://h/a.gguf", "https://h/b.gguf");
            let (a, b) = (board.bar(), board.bar());
            a.on_start(&progress(ua, 0, 0));
            b.on_start(&progress(ub, 0, 0));
            a.on_update(&progress(ua, 10, 100));
            b.on_update(&progress(ub, 20, 200));
            a.on_update(&progress(ua, 50, 100));
            b.on_update(&progress(ub, 200, 200));
            a.on_update(&progress(ua, 60, 100));
            a.on_done(&progress(ua, 60, 100), true);
        });
        assert_eq!(
            out,
            expected(
                "\n\x1b[1A\rDownloading a.gguf ####>                                              10%\x1b[K\x1b[1B\r\n\x1b[1A\rDownloading b.gguf ####>                                              10%\x1b[K\x1b[1B\r\x1b[2A\rDownloading a.gguf ########################>                          50%\x1b[K\x1b[2B\r\x1b[1A\rDownloading b.gguf ################################################# 100%\x1b[K\x1b[1B\r\x1b[2A\rDownloading a.gguf #############################                      60%\x1b[K\x1b[2B\r"
            )
        );
    }

    #[test]
    fn nothing_is_drawn_without_a_tty() {
        assert_eq!(render(false, single), "");
    }
}
