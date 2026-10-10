use std::collections::BTreeMap;

use serde_json::{Map, Value};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Format {
    None,
    Uuid,
    Date,
    Time,
    DateTime,
}

pub struct Property {
    pub name: String,
    pub schema: Node,
    pub required: bool,
}

pub enum Node {
    Any,
    Ref(String),
    AnyOf(Vec<Node>),
    AllOf(Vec<Node>),
    Const(Value),
    Enum(Vec<Value>),
    Null,
    Boolean,
    Number,
    Integer {
        minimum: i64,
        maximum: i64,
    },
    Str {
        pattern: String,
        format: Format,
        min_length: i32,
        max_length: i32,
    },
    Array {
        items: Box<Node>,
        min_items: i32,
        max_items: i32,
    },
    Tuple(Vec<Node>),
    Object {
        properties: Vec<Property>,
        additional_properties: Option<Box<Node>>,
    },
}

const MAX_BUILD_DEPTH: usize = 512;

pub struct Document {
    pub root: Node,
    pub refs: BTreeMap<String, Node>,
}

pub fn parse(schema: &Value) -> Result<Document, String> {
    let mut builder = Builder {
        root: schema,
        refs: BTreeMap::new(),
        depth: 0,
    };
    let root = builder.build_node(schema, "#")?;
    let refs = builder
        .refs
        .into_iter()
        .filter_map(|(key, node)| node.map(|node| (key, node)))
        .collect();
    Ok(Document { root, refs })
}

fn fail<T>(path: &str, msg: &str) -> Result<T, String> {
    Err(format!("JSON schema error at {path}: {msg}"))
}

fn integer(value: &Value) -> Option<i64> {
    let n = value.as_number()?;
    n.as_i64().or_else(|| n.as_u64().map(|u| u as i64))
}

fn stoull(s: &str) -> Option<u64> {
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
    let value = rest[..end].parse::<u64>().ok()?;
    Some(if negative {
        value.wrapping_neg()
    } else {
        value
    })
}

fn float_to_i64(d: f64) -> i64 {
    if d.is_nan() || d >= 9223372036854775808.0 || d < -9223372036854775808.0 {
        i64::MIN
    } else {
        d as i64
    }
}

fn get_count(schema: &Map<String, Value>, key: &str, path: &str, def: i32) -> Result<i32, String> {
    let Some(value) = schema.get(key) else {
        return Ok(def);
    };
    match integer(value) {
        Some(n) if (n as i32) >= 0 => Ok(n as i32),
        _ => fail(path, &format!("{key} must be a non-negative integer")),
    }
}

fn get_bound(
    schema: &Map<String, Value>,
    key: &str,
    path: &str,
    round_up: bool,
) -> Result<i64, String> {
    let value = &schema[key];
    if let Some(n) = integer(value) {
        return Ok(n);
    }
    let Some(d) = value.as_f64() else {
        return fail(path, &format!("{key} must be a number"));
    };
    Ok(float_to_i64(if round_up { d.ceil() } else { d.floor() }))
}

fn get_format(schema: &Map<String, Value>, path: &str) -> Result<Format, String> {
    let Some(value) = schema.get("format") else {
        return Ok(Format::None);
    };
    let Some(format) = value.as_str() else {
        return fail(path, "format must be a string");
    };
    let b = format.as_bytes();
    Ok(match format {
        "date" => Format::Date,
        "time" => Format::Time,
        "date-time" => Format::DateTime,
        "uuid" => Format::Uuid,
        _ if b.len() == 5 && &b[..4] == b"uuid" && (b'1'..=b'5').contains(&b[4]) => Format::Uuid,
        _ => Format::None,
    })
}

fn resolve_ref<'v>(root: &'v Value, reference: &str, path: &str) -> Result<&'v Value, String> {
    let mut target = root;
    let tokens: Vec<&str> = reference[1..].split('/').collect();
    for sel in tokens.iter().skip(1) {
        if let Some(next) = target.as_object().and_then(|o| o.get(*sel)) {
            target = next;
        } else if let Some(items) = target.as_array() {
            let idx = stoull(sel).unwrap_or(items.len() as u64);
            if idx >= items.len() as u64 {
                return fail(
                    path,
                    &format!("cannot resolve $ref {reference}, {sel} is out of range"),
                );
            }
            target = &items[idx as usize];
        } else {
            return fail(
                path,
                &format!("cannot resolve $ref {reference}, {sel} not found"),
            );
        }
    }
    Ok(target)
}

struct Builder<'a> {
    root: &'a Value,
    refs: BTreeMap<String, Option<Node>>,
    depth: usize,
}

impl<'a> Builder<'a> {
    fn build_ref(&mut self, value: &Value, path: &str) -> Result<Node, String> {
        let Some(reference) = value.as_str() else {
            return fail(path, "$ref must be a string");
        };
        if !reference.starts_with("#/") {
            return fail(
                path,
                &format!(
                    "unsupported $ref {reference}, only references into the same document are supported"
                ),
            );
        }
        if !self.refs.contains_key(reference) {
            self.refs.insert(reference.to_string(), None);
            let target = resolve_ref(self.root, reference, path)?;
            let node = self.build_node(target, reference)?;
            self.refs.insert(reference.to_string(), Some(node));
        }
        Ok(Node::Ref(reference.to_string()))
    }

    fn build_alternatives(&mut self, alts: &Value, path: &str) -> Result<Vec<Node>, String> {
        let Some(alts) = alts.as_array() else {
            return fail(path, "must be an array of schemas");
        };
        if alts.is_empty() {
            return fail(path, "must not be empty");
        }
        alts.iter()
            .enumerate()
            .map(|(i, alt)| self.build_node(alt, &format!("{path}/{i}")))
            .collect()
    }

    fn build_object(&mut self, schema: &Map<String, Value>, path: &str) -> Result<Node, String> {
        let mut required: Vec<&str> = Vec::new();
        if let Some(Value::Array(names)) = schema.get("required") {
            required.extend(names.iter().filter_map(Value::as_str));
        }

        let mut properties = Vec::new();
        if let Some(props) = schema.get("properties") {
            let Some(props) = props.as_object() else {
                return fail(path, "properties must be an object");
            };
            for (name, prop) in props {
                let node = self.build_node(prop, &format!("{path}/properties/{name}"))?;
                properties.push(Property {
                    name: name.clone(),
                    schema: node,
                    required: required.contains(&name.as_str()),
                });
            }
        }

        let mut additional_properties = None;
        if let Some(additional) = schema.get("additionalProperties") {
            match additional {
                Value::Bool(true) => additional_properties = Some(Box::new(Node::Any)),
                Value::Bool(false) => {}
                Value::Object(_) => {
                    let node =
                        self.build_node(additional, &format!("{path}/additionalProperties"))?;
                    additional_properties = Some(Box::new(node));
                }
                _ => return fail(path, "additionalProperties must be a boolean or a schema"),
            }
        } else if !schema.contains_key("properties") {
            additional_properties = Some(Box::new(Node::Any));
        }

        Ok(Node::Object {
            properties,
            additional_properties,
        })
    }

    fn build_array(&mut self, schema: &Map<String, Value>, path: &str) -> Result<Node, String> {
        let items_node = if schema.contains_key("items") || schema.contains_key("prefixItems") {
            let key = if schema.contains_key("items") {
                "items"
            } else {
                "prefixItems"
            };
            let items = &schema[key];
            if let Some(items) = items.as_array() {
                let nodes = items
                    .iter()
                    .enumerate()
                    .map(|(i, item)| self.build_node(item, &format!("{path}/{key}/{i}")))
                    .collect::<Result<Vec<_>, _>>()?;
                return Ok(Node::Tuple(nodes));
            }
            self.build_node(items, &format!("{path}/{key}"))?
        } else {
            Node::Any
        };
        Ok(Node::Array {
            items: Box::new(items_node),
            min_items: get_count(schema, "minItems", path, 0)?,
            max_items: get_count(schema, "maxItems", path, -1)?,
        })
    }

    fn build_string(&self, schema: &Map<String, Value>, path: &str) -> Result<Node, String> {
        let mut pattern = String::new();
        if let Some(value) = schema.get("pattern") {
            let Some(p) = value.as_str() else {
                return fail(path, "pattern must be a string");
            };
            pattern = p.to_string();
        }
        Ok(Node::Str {
            pattern,
            format: get_format(schema, path)?,
            min_length: get_count(schema, "minLength", path, 0)?,
            max_length: get_count(schema, "maxLength", path, -1)?,
        })
    }

    fn build_integer(&self, schema: &Map<String, Value>, path: &str) -> Result<Node, String> {
        let mut minimum = i64::MIN;
        let mut maximum = i64::MAX;
        if schema.contains_key("minimum") {
            minimum = get_bound(schema, "minimum", path, true)?;
        } else if schema.contains_key("exclusiveMinimum") {
            minimum = get_bound(schema, "exclusiveMinimum", path, false)?.wrapping_add(1);
        }
        if schema.contains_key("maximum") {
            maximum = get_bound(schema, "maximum", path, false)?;
        } else if schema.contains_key("exclusiveMaximum") {
            maximum = get_bound(schema, "exclusiveMaximum", path, true)?.wrapping_sub(1);
        }
        Ok(Node::Integer { minimum, maximum })
    }

    fn build_node(&mut self, schema: &Value, path: &str) -> Result<Node, String> {
        if self.depth >= MAX_BUILD_DEPTH {
            return Err("schema nesting too deep".to_string());
        }
        self.depth += 1;
        let result = self.build_node_inner(schema, path);
        self.depth -= 1;
        result
    }

    fn build_node_inner(&mut self, schema: &Value, path: &str) -> Result<Node, String> {
        let Some(obj) = schema.as_object() else {
            return fail(path, "schema must be an object");
        };
        if let Some(reference) = obj.get("$ref") {
            return self.build_ref(reference, path);
        }
        if obj.contains_key("oneOf") || obj.contains_key("anyOf") {
            let key = if obj.contains_key("oneOf") {
                "oneOf"
            } else {
                "anyOf"
            };
            return Ok(Node::AnyOf(
                self.build_alternatives(&obj[key], &format!("{path}/{key}"))?,
            ));
        }

        let type_value = obj.get("type").cloned().unwrap_or(Value::Null);
        if let Value::Array(types) = &type_value {
            if types.is_empty() {
                return fail(path, "type must not be empty");
            }
            let mut children = Vec::new();
            for (i, t) in types.iter().enumerate() {
                let mut alt = obj.clone();
                alt.insert("type".to_string(), t.clone());
                children.push(self.build_node(&Value::Object(alt), &format!("{path}/type/{i}"))?);
            }
            return Ok(Node::AnyOf(children));
        }
        if let Some(value) = obj.get("const") {
            return Ok(Node::Const(value.clone()));
        }
        if let Some(values) = obj.get("enum") {
            match values.as_array() {
                Some(values) if !values.is_empty() => return Ok(Node::Enum(values.clone())),
                _ => return fail(path, "enum must be a non-empty array"),
            }
        }
        if !type_value.is_null() && !type_value.is_string() {
            return fail(path, "type must be a string or an array of strings");
        }

        let type_name = type_value.as_str().unwrap_or("");
        let has_properties = obj.contains_key("properties")
            || obj
                .get("additionalProperties")
                .is_some_and(|v| *v != Value::Bool(true));

        if type_name.is_empty() {
            if has_properties {
                return self.build_object(obj, path);
            }
            if let Some(all_of) = obj.get("allOf") {
                return Ok(Node::AllOf(
                    self.build_alternatives(all_of, &format!("{path}/allOf"))?,
                ));
            }
            if obj.contains_key("items") || obj.contains_key("prefixItems") {
                return self.build_array(obj, path);
            }
            if obj.contains_key("pattern")
                || obj.contains_key("minLength")
                || obj.contains_key("maxLength")
                || get_format(obj, path)? != Format::None
            {
                return self.build_string(obj, path);
            }
            return Ok(Node::Any);
        }
        match type_name {
            "object" => {
                if !has_properties && let Some(all_of) = obj.get("allOf") {
                    return Ok(Node::AllOf(
                        self.build_alternatives(all_of, &format!("{path}/allOf"))?,
                    ));
                }
                self.build_object(obj, path)
            }
            "string" => {
                if let Some(all_of) = obj.get("allOf") {
                    return Ok(Node::AllOf(
                        self.build_alternatives(all_of, &format!("{path}/allOf"))?,
                    ));
                }
                self.build_string(obj, path)
            }
            "array" => self.build_array(obj, path),
            "integer" => self.build_integer(obj, path),
            "number" => Ok(Node::Number),
            "boolean" => Ok(Node::Boolean),
            "null" => Ok(Node::Null),
            _ => fail(path, &format!("unrecognized type {type_name}")),
        }
    }
}
