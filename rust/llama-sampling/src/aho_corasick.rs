use std::collections::{BTreeMap, VecDeque};

#[derive(Clone, Default)]
struct Node {
    children: BTreeMap<u32, usize>,
    pattern: Option<usize>,
}

#[derive(Clone)]
pub struct AhoCorasick {
    nodes: Vec<Node>,
    fail: Vec<usize>,
    matches: Vec<Option<usize>>,
}

impl AhoCorasick {
    pub fn new<'a>(patterns: impl IntoIterator<Item = &'a [u32]>) -> Self {
        let mut nodes = vec![Node::default()];
        let mut n_patterns = 0;
        for symbols in patterns {
            let mut current = 0;
            for &ch in symbols {
                current = match nodes[current].children.get(&ch) {
                    Some(&child) => child,
                    None => {
                        nodes.push(Node::default());
                        let child = nodes.len() - 1;
                        nodes[current].children.insert(ch, child);
                        child
                    }
                };
            }
            if nodes[current].pattern.is_none() {
                nodes[current].pattern = Some(n_patterns);
                n_patterns += 1;
            }
        }

        let mut fail = vec![0; nodes.len()];
        let mut order = Vec::with_capacity(nodes.len());
        let mut queue = VecDeque::from([0]);
        while let Some(u) = queue.pop_front() {
            order.push(u);
            for (&ch, &v) in &nodes[u].children {
                if u != 0 {
                    let mut f = fail[u];
                    while f != 0 && !nodes[f].children.contains_key(&ch) {
                        f = fail[f];
                    }
                    fail[v] = match nodes[f].children.get(&ch) {
                        Some(&w) if w != v => w,
                        _ => 0,
                    };
                }
                queue.push_back(v);
            }
        }

        let mut matches = vec![None; nodes.len()];
        for &u in &order {
            matches[u] = nodes[u]
                .pattern
                .or(if u != 0 { matches[fail[u]] } else { None });
        }

        Self {
            nodes,
            fail,
            matches,
        }
    }

    pub fn next(&self, mut state: usize, ch: u32) -> usize {
        while state != 0 && !self.nodes[state].children.contains_key(&ch) {
            state = self.fail[state];
        }
        self.nodes[state].children.get(&ch).copied().unwrap_or(0)
    }

    pub fn match_pattern(&self, state: usize) -> Option<usize> {
        self.matches[state]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed(ac: &AhoCorasick, input: &[u32]) -> Vec<Option<usize>> {
        let mut state = 0;
        input
            .iter()
            .map(|&ch| {
                state = ac.next(state, ch);
                ac.match_pattern(state)
            })
            .collect()
    }

    #[test]
    fn reports_longest_pattern_ending_at_each_position() {
        let patterns: [&[u32]; 3] = [&[2], &[1, 2], &[1, 2]];
        let ac = AhoCorasick::new(patterns);
        assert_eq!(
            feed(&ac, &[1, 2, 3, 2, 1, 1, 2]),
            vec![None, Some(1), None, Some(0), None, None, Some(1)]
        );
    }
}
