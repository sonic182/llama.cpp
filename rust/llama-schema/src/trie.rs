use std::collections::BTreeMap;

pub struct Node {
    pub children: BTreeMap<u32, usize>,
    pub pattern: i32,
}

pub struct Trie {
    pub nodes: Vec<Node>,
    n_patterns: i32,
}

impl Trie {
    pub fn new(words: &[String]) -> Self {
        let mut trie = Trie {
            nodes: vec![Node {
                children: BTreeMap::new(),
                pattern: -1,
            }],
            n_patterns: 0,
        };
        for word in words {
            trie.insert(word);
        }
        trie
    }

    fn insert(&mut self, word: &str) {
        let mut current = 0;
        for ch in word.chars() {
            let ch = ch as u32;
            current = match self.nodes[current].children.get(&ch) {
                Some(&child) => child,
                None => {
                    let child = self.nodes.len();
                    self.nodes.push(Node {
                        children: BTreeMap::new(),
                        pattern: -1,
                    });
                    self.nodes[current].children.insert(ch, child);
                    child
                }
            };
        }
        if self.nodes[current].pattern < 0 {
            self.nodes[current].pattern = self.n_patterns;
            self.n_patterns += 1;
        }
    }
}
