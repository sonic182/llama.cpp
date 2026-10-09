use std::collections::{BTreeMap, HashMap, HashSet};

use crate::dump::{dump, dump_string};
use crate::schema::{Document, Format, Node, Property};
use crate::trie::Trie;

const SPACE_RULE: &str = "| \" \" | \"\\n\"{1,2} [ \\t]{0,20}";
const MAX_PATTERN_DEPTH: i32 = 100;
const MAX_VISIT_DEPTH: usize = 256;
const UNBOUNDED: i32 = i32::MAX;

struct Builtin {
    content: &'static str,
    deps: &'static [&'static str],
}

fn primitive_rule(name: &str) -> Option<Builtin> {
    let (content, deps): (&'static str, &'static [&'static str]) = match name {
        "boolean" => ("(\"true\" | \"false\")", &[]),
        "decimal-part" => ("[0-9]{1,16}", &[]),
        "integral-part" => ("[0] | [1-9] [0-9]{0,15}", &[]),
        "number" => (
            "(\"-\"? integral-part) (\".\" decimal-part)? ([eE] [-+]? integral-part)?",
            &["integral-part", "decimal-part"],
        ),
        "integer" => ("(\"-\"? integral-part)", &["integral-part"]),
        "value" => (
            "object | array | string | number | boolean | null",
            &["object", "array", "string", "number", "boolean", "null"],
        ),
        "object" => (
            "\"{\" space ( string \":\" space value (\",\" space string \":\" space value)* )? space \"}\"",
            &["string", "value"],
        ),
        "array" => (
            "\"[\" space ( value (\",\" space value)* )? space \"]\"",
            &["value"],
        ),
        "uuid" => (
            "\"\\\"\" [0-9a-fA-F]{8} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{12} \"\\\"\"",
            &[],
        ),
        "char" => (
            "[^\"\\\\\\x7F\\x00-\\x1F] | [\\\\] ([\"\\\\bfnrt] | \"u\" [0-9a-fA-F]{4})",
            &[],
        ),
        "string" => ("\"\\\"\" char* \"\\\"\"", &["char"]),
        "null" => ("\"null\"", &[]),
        _ => return None,
    };
    Some(Builtin { content, deps })
}

fn format_rule(name: &str) -> Option<Builtin> {
    let (content, deps): (&'static str, &'static [&'static str]) = match name {
        "date" => (
            "[0-9]{4} \"-\" ( \"0\" [1-9] | \"1\" [0-2] ) \"-\" ( \"0\" [1-9] | [1-2] [0-9] | \"3\" [0-1] )",
            &[],
        ),
        "time" => (
            "([01] [0-9] | \"2\" [0-3]) \":\" [0-5] [0-9] \":\" [0-5] [0-9] ( \".\" [0-9]{3} )? ( \"Z\" | ( \"+\" | \"-\" ) ( [01] [0-9] | \"2\" [0-3] ) \":\" [0-5] [0-9] )",
            &[],
        ),
        "date-time" => ("date \"T\" time", &["date", "time"]),
        "date-string" => ("\"\\\"\" date \"\\\"\"", &["date"]),
        "time-string" => ("\"\\\"\" time \"\\\"\"", &["time"]),
        "date-time-string" => ("\"\\\"\" date-time \"\\\"\"", &["date-time"]),
        _ => return None,
    };
    Some(Builtin { content, deps })
}

fn is_reserved_name(name: &str) -> bool {
    name == "root" || primitive_rule(name).is_some() || format_rule(name).is_some()
}

fn replace_invalid_runs(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut in_run = false;
    for c in input.chars() {
        if c.is_ascii_alphanumeric() || c == '-' {
            in_run = false;
            out.push(c);
        } else if !in_run {
            in_run = true;
            out.push('-');
        }
    }
    out
}

fn format_literal(literal: &str) -> String {
    let mut out = String::with_capacity(literal.len() + 2);
    out.push('"');
    for c in literal.chars() {
        match c {
            '\r' => out.push_str("\\r"),
            '\n' => out.push_str("\\n"),
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn build_repetition(
    item_rule: &str,
    min_items: i32,
    max_items: i32,
    separator_rule: &str,
) -> String {
    let has_max = max_items != UNBOUNDED;

    if max_items == 0 {
        return String::new();
    }
    if min_items == 0 && max_items == 1 {
        return format!("{item_rule}?");
    }

    if separator_rule.is_empty() {
        if min_items == 1 && !has_max {
            return format!("{item_rule}+");
        }
        if min_items == 0 && !has_max {
            return format!("{item_rule}*");
        }
        let max = if has_max {
            max_items.to_string()
        } else {
            String::new()
        };
        return format!("{item_rule}{{{min_items},{max}}}");
    }

    let inner = build_repetition(
        &format!("({separator_rule} {item_rule})"),
        if min_items == 0 { 0 } else { min_items - 1 },
        if has_max { max_items - 1 } else { max_items },
        "",
    );
    let result = format!("{item_rule} {inner}");
    if min_items == 0 {
        format!("({result})?")
    } else {
        result
    }
}

fn digit_range(out: &mut String, from: u8, to: u8) {
    out.push('[');
    out.push(from as char);
    if from != to {
        out.push('-');
        out.push(to as char);
    }
    out.push(']');
}

fn more_digits(out: &mut String, min_digits: i32, max_digits: i32) {
    out.push_str("[0-9]");
    if min_digits == max_digits && min_digits == 1 {
        return;
    }
    out.push('{');
    out.push_str(&min_digits.to_string());
    if max_digits != min_digits {
        out.push(',');
        if max_digits != UNBOUNDED {
            out.push_str(&max_digits.to_string());
        }
    }
    out.push('}');
}

fn uniform_range(out: &mut String, from: &[u8], to: &[u8]) {
    let mut i = 0;
    while i < from.len() && i < to.len() && from[i] == to[i] {
        i += 1;
    }
    if i > 0 {
        out.push('"');
        out.push_str(&String::from_utf8_lossy(&from[..i]));
        out.push('"');
    }
    if i < from.len() && i < to.len() {
        if i > 0 {
            out.push(' ');
        }
        let sub_len = from.len() - i - 1;
        if sub_len > 0 {
            let from_sub = &from[i + 1..];
            let to_sub = &to[i + 1..];
            let sub_zeros = vec![b'0'; sub_len];
            let sub_nines = vec![b'9'; sub_len];

            let mut to_reached = false;
            out.push('(');
            if from_sub == sub_zeros.as_slice() {
                digit_range(out, from[i], to[i].wrapping_sub(1));
                out.push(' ');
                more_digits(out, sub_len as i32, sub_len as i32);
            } else {
                out.push('[');
                out.push(from[i] as char);
                out.push_str("] (");
                uniform_range(out, from_sub, &sub_nines);
                out.push(')');
                if (from[i] as i32) < to[i] as i32 - 1 {
                    out.push_str(" | ");
                    if to_sub == sub_nines.as_slice() {
                        digit_range(out, from[i] + 1, to[i]);
                        to_reached = true;
                    } else {
                        digit_range(out, from[i] + 1, to[i] - 1);
                    }
                    out.push(' ');
                    more_digits(out, sub_len as i32, sub_len as i32);
                }
            }
            if !to_reached {
                out.push_str(" | ");
                digit_range(out, to[i], to[i]);
                out.push(' ');
                uniform_range(out, &sub_zeros, to_sub);
            }
            out.push(')');
        } else {
            out.push('[');
            out.push(from[i] as char);
            out.push('-');
            out.push(to[i] as char);
            out.push(']');
        }
    }
}

fn repeat9(n: usize) -> Vec<u8> {
    vec![b'9'; n]
}

fn build_min_max_int(
    min_value: i64,
    max_value: i64,
    out: &mut String,
    decimals_left: i32,
    top_level: bool,
) -> Result<(), String> {
    let mut min_value = min_value;
    let has_min = min_value != i64::MIN;
    let has_max = max_value != i64::MAX;

    if has_min && has_max {
        if min_value < 0 && max_value < 0 {
            out.push_str("\"-\" (");
            build_min_max_int(
                max_value.wrapping_neg(),
                min_value.wrapping_neg(),
                out,
                decimals_left,
                true,
            )?;
            out.push(')');
            return Ok(());
        }

        if min_value < 0 {
            out.push_str("\"-\" (");
            build_min_max_int(0, min_value.wrapping_neg(), out, decimals_left, true)?;
            out.push_str(") | ");
            min_value = 0;
        }

        let mut min_s = min_value.to_string().into_bytes();
        let max_s = max_value.to_string().into_bytes();
        let min_digits = min_s.len();
        let max_digits = max_s.len();

        for digits in min_digits..max_digits {
            uniform_range(out, &min_s, &repeat9(digits));
            min_s = format!("1{}", "0".repeat(digits)).into_bytes();
            out.push_str(" | ");
        }
        uniform_range(out, &min_s, &max_s);
        return Ok(());
    }

    let less_decimals = (decimals_left - 1).max(1);

    if has_min {
        if min_value < 0 {
            out.push_str("\"-\" (");
            build_min_max_int(
                i64::MIN,
                min_value.wrapping_neg(),
                out,
                decimals_left,
                false,
            )?;
            out.push_str(") | [0] | [1-9] ");
            more_digits(out, 0, decimals_left - 1);
        } else if min_value == 0 {
            if top_level {
                out.push_str("[0] | [1-9] ");
                more_digits(out, 0, less_decimals);
            } else {
                more_digits(out, 1, decimals_left);
            }
        } else if min_value <= 9 {
            let c = b'0' + min_value as u8;
            let range_start = if top_level { b'1' } else { b'0' };
            if c > range_start {
                digit_range(out, range_start, c - 1);
                out.push(' ');
                more_digits(out, 1, less_decimals);
                out.push_str(" | ");
            }
            digit_range(out, c, b'9');
            out.push(' ');
            more_digits(out, 0, less_decimals);
        } else {
            let min_s = min_value.to_string().into_bytes();
            let len = min_s.len() as i32;
            let c = min_s[0];

            if c > b'1' {
                digit_range(out, if top_level { b'1' } else { b'0' }, c - 1);
                out.push(' ');
                more_digits(out, len, less_decimals);
                out.push_str(" | ");
            }
            digit_range(out, c, c);
            out.push_str(" (");
            let rest = String::from_utf8_lossy(&min_s[1..])
                .parse::<i64>()
                .unwrap_or(0);
            build_min_max_int(rest, i64::MAX, out, less_decimals, false)?;
            out.push(')');
            if c < b'9' {
                out.push_str(" | ");
                digit_range(out, c + 1, b'9');
                out.push(' ');
                more_digits(out, len - 1, less_decimals);
            }
        }
        return Ok(());
    }

    if has_max {
        if max_value >= 0 {
            if top_level {
                out.push_str("\"-\" [1-9] ");
                more_digits(out, 0, less_decimals);
                out.push_str(" | ");
            }
            build_min_max_int(0, max_value, out, decimals_left, true)?;
        } else {
            out.push_str("\"-\" (");
            build_min_max_int(
                max_value.wrapping_neg(),
                i64::MAX,
                out,
                decimals_left,
                false,
            )?;
            out.push(')');
        }
        return Ok(());
    }

    Err(
        "JSON schema conversion failed:\nAt least one of min_value or max_value must be set"
            .to_string(),
    )
}

fn gbnf_escape_length(pattern: &[char], pos: usize) -> usize {
    if pos + 1 >= pattern.len() || pattern[pos] != '\\' {
        return 0;
    }
    let n_hex = match pattern[pos + 1] {
        'x' => 2,
        'u' => 4,
        'U' => 8,
        't' | 'r' | 'n' | '\\' | '"' | '[' | ']' | '-' => return 2,
        _ => return 0,
    };
    if pos + 2 + n_hex > pattern.len() {
        return 0;
    }
    if !pattern[pos + 2..pos + 2 + n_hex]
        .iter()
        .all(|h| h.is_ascii_hexdigit())
    {
        return 0;
    }
    2 + n_hex
}

fn stoi(s: &str) -> Option<i32> {
    let s = s.trim_start_matches([' ', '\t', '\n', '\u{b}', '\u{c}', '\r']);
    let (negative, rest) = match s.as_bytes().first() {
        Some(b'-') => (true, &s[1..]),
        Some(b'+') => (false, &s[1..]),
        _ => (false, s),
    };
    let end = rest
        .bytes()
        .position(|b| !b.is_ascii_digit())
        .unwrap_or(rest.len());
    if end == 0 {
        return None;
    }
    let magnitude = rest[..end].parse::<i64>().ok()?;
    let value = if negative { -magnitude } else { magnitude };
    i32::try_from(value).ok()
}

enum PatternError {
    Unsupported(String),
    Invalid(String),
}

fn unsupported<T>(msg: impl Into<String>) -> Result<T, PatternError> {
    Err(PatternError::Unsupported(msg.into()))
}

fn invalid<T>(msg: impl Into<String>) -> Result<T, PatternError> {
    Err(PatternError::Invalid(msg.into()))
}

type LiteralOrRule = (String, bool);

fn is_non_literal(c: char) -> bool {
    matches!(
        c,
        '|' | '.' | '(' | ')' | '[' | ']' | '{' | '}' | '*' | '+' | '?' | '^' | '$'
    )
}

fn escaped_in_regexps_but_not_in_literals(c: char) -> bool {
    matches!(
        c,
        '^' | '$' | '.' | '[' | ']' | '(' | ')' | '|' | '{' | '}' | '*' | '+' | '?'
    )
}

fn to_rule(item: &LiteralOrRule) -> String {
    if item.1 {
        format!("\"{}\"", item.0)
    } else {
        item.0.clone()
    }
}

fn join_seq(seq: &[LiteralOrRule]) -> LiteralOrRule {
    let mut merged: Vec<LiteralOrRule> = Vec::new();
    let mut literal = String::new();
    for item in seq {
        if item.1 {
            literal.push_str(&item.0);
        } else {
            if !literal.is_empty() {
                merged.push((std::mem::take(&mut literal), true));
            }
            merged.push(item.clone());
        }
    }
    if !literal.is_empty() {
        merged.push((literal, true));
    }
    let rules: Vec<String> = merged.iter().map(to_rule).collect();
    (rules.join(" "), false)
}

fn chars_to_string(chars: &[char]) -> String {
    chars.iter().collect()
}

struct PatternParser<'c, 'd> {
    conv: &'c mut Converter<'d>,
    name: &'c str,
    sub_pattern: &'c [char],
    i: usize,
    paren_depth: i32,
    sub_rule_ids: HashMap<String, String>,
}

impl PatternParser<'_, '_> {
    fn transform(&mut self) -> Result<LiteralOrRule, PatternError> {
        let mut seq: Vec<LiteralOrRule> = Vec::new();
        let length = self.sub_pattern.len();

        while self.i < length {
            let c = self.sub_pattern[self.i];
            if c == '.' {
                let rule = if self.conv.dotall {
                    "[\\U00000000-\\U0010FFFF]"
                } else {
                    "[^\\x0A\\x0D]"
                };
                seq.push((self.conv.add_rule("dot", rule), false));
                self.i += 1;
            } else if c == '(' {
                self.i += 1;
                if self.i < length && self.sub_pattern[self.i] == '?' {
                    if self.i + 1 < length && self.sub_pattern[self.i + 1] == ':' {
                        self.i += 2;
                    } else {
                        return unsupported("unsupported group syntax");
                    }
                }
                self.paren_depth += 1;
                if self.paren_depth > MAX_PATTERN_DEPTH {
                    return unsupported("pattern nesting too deep");
                }
                let inner = self.transform()?;
                seq.push((format!("({})", to_rule(&inner)), false));
            } else if c == ')' {
                self.i += 1;
                if self.paren_depth == 0 {
                    return invalid("unbalanced parentheses");
                }
                self.paren_depth -= 1;
                return Ok(join_seq(&seq));
            } else if c == '^' || c == '$' {
                return unsupported("anchor inside the pattern");
            } else if c == '[' {
                let mut square_brackets = String::from(c);
                self.i += 1;
                while self.i < length && self.sub_pattern[self.i] != ']' {
                    if self.sub_pattern[self.i] == '\\' {
                        let escape_length = gbnf_escape_length(self.sub_pattern, self.i);
                        if escape_length == 0 {
                            let end = (self.i + 2).min(length);
                            return unsupported(format!(
                                "unsupported escape in character class: {}",
                                chars_to_string(&self.sub_pattern[self.i..end])
                            ));
                        }
                        square_brackets.push_str(&chars_to_string(
                            &self.sub_pattern[self.i..self.i + escape_length],
                        ));
                        self.i += escape_length;
                    } else {
                        square_brackets.push(self.sub_pattern[self.i]);
                        self.i += 1;
                    }
                }
                if self.i >= length {
                    return invalid("unterminated character class");
                }
                square_brackets.push(']');
                self.i += 1;
                seq.push((square_brackets, false));
            } else if c == '|' {
                seq.push(("|".to_string(), false));
                self.i += 1;
            } else if c == '*' || c == '+' || c == '?' {
                let Some(last) = seq.last_mut() else {
                    return invalid("nothing to repeat");
                };
                let mut rule = to_rule(last);
                rule.push(c);
                *last = (rule, false);
                self.i += 1;
            } else if c == '{' {
                let mut curly_brackets = String::from(c);
                self.i += 1;
                while self.i < length && self.sub_pattern[self.i] != '}' {
                    curly_brackets.push(self.sub_pattern[self.i]);
                    self.i += 1;
                }
                if self.i >= length {
                    return unsupported("unterminated curly brackets");
                }
                curly_brackets.push('}');
                self.i += 1;
                let inner = &curly_brackets[1..curly_brackets.len() - 1];
                let nums: Vec<&str> = inner.split(',').collect();
                let mut min_times: i32 = 0;
                let mut max_times: i32 = UNBOUNDED;
                if nums.len() != 1 && nums.len() != 2 {
                    return unsupported("wrong number of values in curly brackets");
                }
                if nums.len() == 1 {
                    let Some(n) = stoi(nums[0]) else {
                        return unsupported("invalid number in curly brackets");
                    };
                    min_times = n;
                    max_times = n;
                } else {
                    if !nums[0].is_empty() {
                        let Some(n) = stoi(nums[0]) else {
                            return unsupported("invalid number in curly brackets");
                        };
                        min_times = n;
                    }
                    if !nums[1].is_empty() {
                        let Some(n) = stoi(nums[1]) else {
                            return unsupported("invalid number in curly brackets");
                        };
                        max_times = n;
                    }
                }
                let Some(last) = seq.last_mut() else {
                    return invalid("nothing to repeat");
                };
                let sub_is_literal = last.1;
                if !sub_is_literal {
                    if !self.sub_rule_ids.contains_key(&last.0) {
                        self.sub_rule_ids.insert(last.0.clone(), String::new());
                    }
                    let count = self.sub_rule_ids.len();
                    if self.sub_rule_ids[&last.0].is_empty() {
                        let id = self
                            .conv
                            .add_rule(&format!("{}-{}", self.name, count), &last.0);
                        self.sub_rule_ids.insert(last.0.clone(), id);
                    }
                    last.0 = self.sub_rule_ids[&last.0].clone();
                }
                let item = if sub_is_literal {
                    format!("\"{}\"", last.0)
                } else {
                    last.0.clone()
                };
                last.0 = build_repetition(&item, min_times, max_times, "");
                last.1 = false;
            } else {
                let mut literal = String::new();
                while self.i < length {
                    let ch = self.sub_pattern[self.i];
                    if ch == '\\' {
                        if self.i == length - 1 {
                            return invalid("trailing backslash");
                        }
                        let next = self.sub_pattern[self.i + 1];
                        if escaped_in_regexps_but_not_in_literals(next) {
                            self.i += 1;
                            literal.push(self.sub_pattern[self.i]);
                            self.i += 1;
                        } else {
                            let escape_length = gbnf_escape_length(self.sub_pattern, self.i);
                            if escape_length == 0 {
                                return unsupported(format!(
                                    "unsupported escape: {}",
                                    chars_to_string(&self.sub_pattern[self.i..self.i + 2])
                                ));
                            }
                            literal.push_str(&chars_to_string(
                                &self.sub_pattern[self.i..self.i + escape_length],
                            ));
                            self.i += escape_length;
                        }
                    } else if ch == '"' {
                        literal.push_str("\\\"");
                        self.i += 1;
                    } else if !is_non_literal(ch)
                        && (self.i == length - 1
                            || literal.is_empty()
                            || self.sub_pattern[self.i + 1] == '.'
                            || !is_non_literal(self.sub_pattern[self.i + 1]))
                    {
                        literal.push(ch);
                        self.i += 1;
                    } else {
                        break;
                    }
                }
                if literal.is_empty() {
                    return unsupported(format!("unsupported character: {c}"));
                }
                seq.push((literal, true));
            }
        }
        Ok(join_seq(&seq))
    }
}

struct Components<'d> {
    doc: &'d Document,
    required: HashSet<&'d str>,
    properties: Vec<(&'d str, &'d Node)>,
    enum_values: BTreeMap<String, usize>,
    too_deep: bool,
}

impl<'d> Components<'d> {
    fn add(&mut self, comp: &'d Node, is_required: bool, depth: usize) {
        if depth >= MAX_VISIT_DEPTH {
            self.too_deep = true;
            return;
        }
        match comp {
            Node::Ref(reference) => {
                if let Some(target) = self.doc.refs.get(reference) {
                    self.add(target, is_required, depth + 1);
                }
            }
            Node::Object {
                properties: props, ..
            } => {
                for Property { name, schema, .. } in props {
                    self.properties.push((name, schema));
                    if is_required {
                        self.required.insert(name);
                    }
                }
            }
            Node::Enum(values) => {
                for v in values {
                    *self
                        .enum_values
                        .entry(format_literal(&dump(v)))
                        .or_insert(0) += 1;
                }
            }
            _ => {}
        }
    }
}

pub struct Converter<'d> {
    doc: &'d Document,
    dotall: bool,
    rules: BTreeMap<String, String>,
    refs_being_resolved: HashSet<String>,
    depth: usize,
    errors: Vec<String>,
    warnings: Vec<String>,
}

pub struct Output {
    pub grammar: String,
    pub warnings: Vec<String>,
}

pub fn convert(doc: &Document) -> Result<Output, String> {
    let mut converter = Converter::new(doc, false);
    converter.visit(&doc.root, "")?;
    converter.finish()
}

impl<'d> Converter<'d> {
    fn new(doc: &'d Document, dotall: bool) -> Self {
        let mut rules = BTreeMap::new();
        rules.insert("space".to_string(), SPACE_RULE.to_string());
        Converter {
            doc,
            dotall,
            rules,
            refs_being_resolved: HashSet::new(),
            depth: 0,
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    fn finish(self) -> Result<Output, String> {
        if !self.errors.is_empty() {
            return Err(format!(
                "JSON schema conversion failed:\n{}",
                self.errors.join("\n")
            ));
        }
        let mut grammar = String::new();
        for (name, rule) in &self.rules {
            grammar.push_str(name);
            grammar.push_str(" ::= ");
            grammar.push_str(rule);
            grammar.push('\n');
        }
        Ok(Output {
            grammar,
            warnings: self.warnings,
        })
    }

    fn add_rule(&mut self, name: &str, rule: &str) -> String {
        let esc_name = replace_invalid_runs(name);
        match self.rules.get(&esc_name) {
            None => {
                self.rules.insert(esc_name.clone(), rule.to_string());
                return esc_name;
            }
            Some(existing) if existing == rule => return esc_name,
            Some(_) => {}
        }
        let mut i = 0;
        while self
            .rules
            .get(&format!("{esc_name}{i}"))
            .is_some_and(|existing| existing != rule)
        {
            i += 1;
        }
        let key = format!("{esc_name}{i}");
        self.rules.insert(key.clone(), rule.to_string());
        key
    }

    fn add_primitive(&mut self, name: &str, rule: &Builtin) -> String {
        let n = self.add_rule(name, rule.content);
        for dep in rule.deps {
            let Some(dep_rule) = primitive_rule(dep).or_else(|| format_rule(dep)) else {
                self.errors.push(format!("Rule {dep} not known"));
                continue;
            };
            if !self.rules.contains_key(*dep) {
                self.add_primitive(dep, &dep_rule);
            }
        }
        n
    }

    fn primitive(&mut self, name: &str, type_name: &str) -> String {
        match primitive_rule(type_name) {
            Some(rule) => self.add_primitive(name, &rule),
            None => String::new(),
        }
    }

    fn visit_primitive(&mut self, rule_name: &str, type_name: &str) -> String {
        let name = if rule_name == "root" {
            "root"
        } else {
            type_name
        };
        self.primitive(name, type_name)
    }

    fn generate_union_rule(&mut self, name: &str, alts: &'d [Node]) -> Result<String, String> {
        let mut rules = Vec::with_capacity(alts.len());
        for (i, alt) in alts.iter().enumerate() {
            let prefix = if name.is_empty() { "alternative-" } else { "-" };
            rules.push(self.visit(alt, &format!("{name}{prefix}{i}"))?);
        }
        Ok(rules.join(" | "))
    }

    fn visit_pattern(&mut self, pattern: &str, name: &str) -> String {
        let snapshot = self.rules.clone();
        match self.pattern_to_rule(pattern, name) {
            Ok(rule) => rule,
            Err(PatternError::Unsupported(err)) => {
                self.rules = snapshot;
                self.warnings.push(format!(
                    "pattern {pattern} is not supported ({err}), accepting any string"
                ));
                let string = self.primitive("string", "string");
                self.add_rule(name, &string)
            }
            Err(PatternError::Invalid(err)) => {
                self.rules = snapshot;
                self.errors
                    .push(format!("Invalid pattern {pattern}: {err}"));
                String::new()
            }
        }
    }

    fn pattern_to_rule(&mut self, pattern: &str, name: &str) -> Result<String, PatternError> {
        let chars: Vec<char> = pattern.chars().collect();
        if chars.len() < 2 || chars[0] != '^' || chars[chars.len() - 1] != '$' {
            return unsupported("not anchored with '^' and '$'");
        }
        let mut parser = PatternParser {
            conv: self,
            name,
            sub_pattern: &chars[1..chars.len() - 1],
            i: 0,
            paren_depth: 0,
            sub_rule_ids: HashMap::new(),
        };
        let result = parser.transform()?;
        let rule = to_rule(&result);
        if parser.paren_depth != 0 {
            return invalid("unbalanced parentheses");
        }
        Ok(self.add_rule(name, &format!("\"\\\"\" ({rule}) \"\\\"\"")))
    }

    fn not_strings(&mut self, strings: &[String]) -> String {
        let trie = Trie::new(strings);
        let char_rule = self.primitive("char", "char");
        let mut out = String::from("[\"] ( ");

        fn walk(trie: &Trie, idx: usize, out: &mut String, char_rule: &str) {
            let node = &trie.nodes[idx];
            let mut rejects = String::new();
            let mut first = true;
            for (&cpt, &child) in &node.children {
                let c = char::from_u32(cpt).unwrap_or('\u{fffd}');
                rejects.push(c);
                if first {
                    first = false;
                } else {
                    out.push_str(" | ");
                }
                out.push('[');
                out.push(c);
                out.push(']');
                if !trie.nodes[child].children.is_empty() {
                    out.push_str(" (");
                    walk(trie, child, out, char_rule);
                    out.push(')');
                } else {
                    out.push(' ');
                    out.push_str(char_rule);
                    out.push('+');
                }
            }
            if !node.children.is_empty() {
                out.push_str(" | [^\"");
                out.push_str(&rejects);
                out.push_str("] ");
                out.push_str(char_rule);
                out.push('*');
            }
        }
        walk(&trie, 0, &mut out, &char_rule);

        out.push_str(" )");
        if trie.nodes[0].pattern < 0 {
            out.push('?');
        }
        out.push_str(" [\"]");
        out
    }

    fn resolve_ref(&mut self, reference: &str) -> Result<String, String> {
        let fragment = match reference.find('#') {
            Some(pos) => &reference[pos + 1..],
            None => reference,
        };
        let mut ref_name = format!("ref{}", replace_invalid_runs(fragment));
        if !self.rules.contains_key(&ref_name) && !self.refs_being_resolved.contains(reference) {
            let doc = self.doc;
            let Some(target) = doc.refs.get(reference) else {
                self.errors.push(format!("Unresolved $ref {reference}"));
                return Ok(String::new());
            };
            self.refs_being_resolved.insert(reference.to_string());
            ref_name = self.visit(target, &ref_name)?;
            self.refs_being_resolved.remove(reference);
        }
        Ok(ref_name)
    }

    fn get_recursive_refs(
        &mut self,
        kv_rules: &HashMap<String, String>,
        ks: &[String],
        first_is_optional: bool,
        name: &str,
    ) -> String {
        if ks.is_empty() {
            return String::new();
        }
        let k = &ks[0];
        let kv_rule_name = kv_rules.get(k).cloned().unwrap_or_default();
        let comma_ref = format!("( \",\" space {kv_rule_name} )");
        let mut res = if first_is_optional {
            format!("{comma_ref}{}", if k == "*" { "*" } else { "?" })
        } else if k == "*" {
            format!("{kv_rule_name} {comma_ref}*")
        } else {
            kv_rule_name.clone()
        };
        if ks.len() > 1 {
            let sep = if name.is_empty() { "" } else { "-" };
            let rest = self.get_recursive_refs(kv_rules, &ks[1..], true, name);
            let added = self.add_rule(&format!("{name}{sep}{k}-rest"), &rest);
            res.push(' ');
            res.push_str(&added);
        }
        res
    }

    fn build_object_rule(
        &mut self,
        properties: &[(&'d str, &'d Node)],
        required: &HashSet<&str>,
        name: &str,
        additional_properties: Option<&'d Node>,
    ) -> Result<String, String> {
        let sep = if name.is_empty() { "" } else { "-" };
        let mut required_props: Vec<String> = Vec::new();
        let mut optional_props: Vec<String> = Vec::new();
        let mut prop_kv_rule_names: HashMap<String, String> = HashMap::new();
        let mut prop_names: Vec<String> = Vec::new();
        for &(prop_name, prop_schema) in properties {
            let prop_rule_name = self.visit(prop_schema, &format!("{name}{sep}{prop_name}"))?;
            let kv = self.add_rule(
                &format!("{name}{sep}{prop_name}-kv"),
                &format!(
                    "{} space \":\" space {}",
                    format_literal(&dump_string(prop_name)),
                    prop_rule_name
                ),
            );
            prop_kv_rule_names.insert(prop_name.to_string(), kv);
            if required.contains(prop_name) {
                required_props.push(prop_name.to_string());
            } else {
                optional_props.push(prop_name.to_string());
            }
            prop_names.push(prop_name.to_string());
        }
        if let Some(additional) = additional_properties {
            let sub_name = format!("{name}{sep}additional");
            let value_rule = if !matches!(additional, Node::Any) {
                self.visit(additional, &format!("{sub_name}-value"))?
            } else {
                self.primitive("value", "value")
            };

            let key_rule = if prop_names.is_empty() {
                self.primitive("string", "string")
            } else {
                let not = self.not_strings(&prop_names);
                self.add_rule(&format!("{sub_name}-k"), &not)
            };
            let kv_rule = self.add_rule(
                &format!("{sub_name}-kv"),
                &format!("{key_rule} \":\" space {value_rule}"),
            );
            prop_kv_rule_names.insert("*".to_string(), kv_rule);
            optional_props.push("*".to_string());
        }

        if required_props.is_empty() && optional_props.is_empty() {
            return Ok("\"{\" space \"}\"".to_string());
        }

        let mut rule = String::from("\"{\" space ");
        for (i, prop) in required_props.iter().enumerate() {
            if i > 0 {
                rule.push_str(" \",\" space ");
            }
            rule.push_str(prop_kv_rule_names.get(prop).map_or("", String::as_str));
        }

        if !optional_props.is_empty() {
            rule.push_str(" (");
            if !required_props.is_empty() {
                rule.push_str(" \",\" space ( ");
            }
            for i in 0..optional_props.len() {
                if i > 0 {
                    rule.push_str(" | ");
                }
                let refs =
                    self.get_recursive_refs(&prop_kv_rule_names, &optional_props[i..], false, name);
                rule.push_str(&refs);
            }
            if !required_props.is_empty() {
                rule.push_str(" )");
            }
            rule.push_str(" )?");
        }

        rule.push_str(" space \"}\"");
        Ok(rule)
    }

    fn visit_all_of(
        &mut self,
        children: &'d [Node],
        name: &str,
        rule_name: &str,
    ) -> Result<String, String> {
        let mut components = Components {
            doc: self.doc,
            required: HashSet::new(),
            properties: Vec::new(),
            enum_values: BTreeMap::new(),
            too_deep: false,
        };
        for child in children {
            if let Node::AnyOf(alts) = child {
                for alt in alts {
                    components.add(alt, false, 0);
                }
            } else {
                components.add(child, true, 0);
            }
        }
        if components.too_deep {
            return Err("JSON schema conversion failed:\nschema nesting too deep".to_string());
        }
        if !components.enum_values.is_empty() {
            let intersection: Vec<String> = components
                .enum_values
                .iter()
                .filter(|(_, count)| **count == children.len())
                .map(|(value, _)| value.clone())
                .collect();
            if !intersection.is_empty() {
                return Ok(self.add_rule(rule_name, &format!("({})", intersection.join(" | "))));
            }
        }
        let object =
            self.build_object_rule(&components.properties, &components.required, name, None)?;
        Ok(self.add_rule(rule_name, &object))
    }

    fn visit(&mut self, schema: &'d Node, name: &str) -> Result<String, String> {
        if self.depth >= MAX_VISIT_DEPTH {
            return Err("JSON schema conversion failed:\nschema nesting too deep".to_string());
        }
        self.depth += 1;
        let result = self.visit_node(schema, name);
        self.depth -= 1;
        result
    }

    fn visit_node(&mut self, schema: &'d Node, name: &str) -> Result<String, String> {
        let rule_name = if is_reserved_name(name) {
            format!("{name}-")
        } else if name.is_empty() {
            "root".to_string()
        } else {
            name.to_string()
        };
        let sub_name = if name.is_empty() {
            String::new()
        } else {
            format!("{name}-")
        };

        match schema {
            Node::Ref(reference) => {
                let resolved = self.resolve_ref(reference)?;
                Ok(self.add_rule(&rule_name, &resolved))
            }
            Node::AnyOf(children) => {
                let union = self.generate_union_rule(name, children)?;
                Ok(self.add_rule(&rule_name, &union))
            }
            Node::AllOf(children) => self.visit_all_of(children, name, &rule_name),
            Node::Const(value) => Ok(self.add_rule(&rule_name, &format_literal(&dump(value)))),
            Node::Enum(values) => {
                let literals: Vec<String> =
                    values.iter().map(|v| format_literal(&dump(v))).collect();
                Ok(self.add_rule(&rule_name, &format!("({})", literals.join(" | "))))
            }
            Node::Object {
                properties,
                additional_properties,
            } => {
                if properties.is_empty()
                    && additional_properties
                        .as_deref()
                        .is_some_and(|a| matches!(a, Node::Any))
                {
                    let object = self.primitive("object", "object");
                    return Ok(self.add_rule(&rule_name, &object));
                }
                let props: Vec<(&'d str, &'d Node)> = properties
                    .iter()
                    .map(|p| (p.name.as_str(), &p.schema))
                    .collect();
                let required: HashSet<&str> = properties
                    .iter()
                    .filter(|p| p.required)
                    .map(|p| p.name.as_str())
                    .collect();
                let object = self.build_object_rule(
                    &props,
                    &required,
                    name,
                    additional_properties.as_deref(),
                )?;
                Ok(self.add_rule(&rule_name, &object))
            }
            Node::Tuple(items) => {
                let mut rule = String::from("\"[\" space ");
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        rule.push_str(" \",\" space ");
                    }
                    rule.push_str(&self.visit(item, &format!("{sub_name}tuple-{i}"))?);
                }
                rule.push_str(" space \"]\"");
                Ok(self.add_rule(&rule_name, &rule))
            }
            Node::Array {
                items,
                min_items,
                max_items,
            } => {
                if matches!(**items, Node::Any) && *min_items == 0 && *max_items < 0 {
                    return Ok(self.visit_primitive(&rule_name, "array"));
                }
                let item_rule_name = self.visit(items, &format!("{sub_name}item"))?;
                let max = if *max_items < 0 {
                    UNBOUNDED
                } else {
                    *max_items
                };
                let repetition = build_repetition(&item_rule_name, *min_items, max, "\",\" space");
                Ok(self.add_rule(&rule_name, &format!("\"[\" space {repetition} space \"]\"")))
            }
            Node::Str {
                pattern,
                format,
                min_length,
                max_length,
            } => {
                if !pattern.is_empty() {
                    return Ok(self.visit_pattern(pattern, &rule_name));
                }
                if *format == Format::Uuid {
                    return Ok(self.visit_primitive(&rule_name, "uuid"));
                }
                if *format != Format::None {
                    let prim_name = match format {
                        Format::Date => "date-string",
                        Format::Time => "time-string",
                        _ => "date-time-string",
                    };
                    let Some(rule) = format_rule(prim_name) else {
                        return Ok(String::new());
                    };
                    let prim = self.add_primitive(prim_name, &rule);
                    return Ok(self.add_rule(&rule_name, &prim));
                }
                if *min_length > 0 || *max_length >= 0 {
                    let char_rule = self.primitive("char", "char");
                    let max = if *max_length < 0 {
                        UNBOUNDED
                    } else {
                        *max_length
                    };
                    let repetition = build_repetition(&char_rule, *min_length, max, "");
                    return Ok(
                        self.add_rule(&rule_name, &format!("\"\\\"\" {repetition} \"\\\"\""))
                    );
                }
                Ok(self.visit_primitive(&rule_name, "string"))
            }
            Node::Integer { minimum, maximum } => {
                if *minimum == i64::MIN && *maximum == i64::MAX {
                    return Ok(self.visit_primitive(&rule_name, "integer"));
                }
                let mut out = String::from("(");
                build_min_max_int(*minimum, *maximum, &mut out, 16, true)?;
                out.push(')');
                Ok(self.add_rule(&rule_name, &out))
            }
            Node::Number => Ok(self.visit_primitive(&rule_name, "number")),
            Node::Boolean => Ok(self.visit_primitive(&rule_name, "boolean")),
            Node::Null => Ok(self.visit_primitive(&rule_name, "null")),
            Node::Any => {
                let value = self.primitive("value", "value");
                Ok(self.add_rule(&rule_name, &value))
            }
        }
    }
}
