#![allow(clippy::missing_safety_doc)]

mod abi;
mod convert;
mod dump;
mod grisu;
mod grisu_table;
mod schema;
mod trie;

pub use abi::{Buf, SchemaResult};
pub use convert::Output;

pub fn json_schema_to_grammar(schema: &str) -> Result<Output, String> {
    let value: serde_json::Value =
        serde_json::from_str(schema).map_err(|e| format!("JSON schema conversion failed:\n{e}"))?;
    let doc = schema::parse(&value).map_err(|e| format!("JSON schema conversion failed:\n{e}"))?;
    convert::convert(&doc)
}
