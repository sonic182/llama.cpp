use llama_schema::json_schema_to_grammar;
use serde_json::Value;

fn normalize(text: &str) -> String {
    text.trim_matches([' ', '\n', '\r', '\t'])
        .split('\n')
        .map(|line| line.trim_start_matches([' ', '\t']))
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn cases_extracted_from_the_cpp_tests() {
    let cases: Vec<Value> = serde_json::from_str(include_str!("golden/cases.json")).unwrap();
    assert!(cases.len() >= 80);
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let schema = case["schema"].as_str().unwrap();
        let result = json_schema_to_grammar(schema);
        if case["success"].as_bool().unwrap() {
            let output = result.unwrap_or_else(|e| panic!("{name}: {e}"));
            let expected = case["grammar"].as_str().unwrap();
            assert_eq!(
                normalize(&output.grammar),
                normalize(expected),
                "{name}\n{schema}"
            );
        } else {
            let error = result
                .err()
                .unwrap_or_else(|| panic!("{name}: expected a failure"));
            assert!(
                error.starts_with("JSON schema conversion failed:\n"),
                "{name}: {error}"
            );
        }
    }
}

fn grammar(schema: &str) -> String {
    json_schema_to_grammar(schema).unwrap().grammar
}

#[test]
fn constants_are_dumped_like_nlohmann() {
    let text = grammar(
        r#"{"enum": [1.0, 1e20, 0.00001, 0.5, -0.0, 1.5e-7, 123456789012345678901234567890, "a/b\u007f\u0001é\n", [1, {"k": null}]]}"#,
    );
    let root = text.lines().find(|l| l.starts_with("root ::=")).unwrap();
    let expected = [
        r#"root ::= ("1.0" | "1e+20" | "1e-05" | "0.5" | "-0.0" | "1.5e-07" | "1.2345678901234568e+29" | "#,
        "\"\\\"a/b\u{7f}\\\\u0001é\\\\n\\\"\"",
        r#" | "[1,{\"k\":null}]")"#,
    ]
    .concat();
    assert_eq!(root, expected);
}

#[test]
fn invalid_json_is_an_error() {
    let error = json_schema_to_grammar("{").err().unwrap();
    assert!(error.starts_with("JSON schema conversion failed:\n"));
}

#[test]
fn cycles_through_all_of_are_an_error_instead_of_a_stack_overflow() {
    let schema = r##"{"allOf": [{"$ref": "#/$defs/d"}], "$defs": {"d": {"type": "object",
        "properties": {"root": {"allOf": [{"$ref": "#/$defs/d"}]}}, "required": []}}}"##;
    let result = std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(move || json_schema_to_grammar(schema).err())
        .unwrap()
        .join()
        .unwrap();
    assert!(result.unwrap().ends_with("schema nesting too deep"));
}

#[test]
fn deep_but_finite_schemas_still_convert() {
    let mut schema = String::from(r#"{"type": "string"}"#);
    for _ in 0..60 {
        schema = format!(r#"{{"type": "array", "items": {schema}}}"#);
    }
    let text = std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(move || grammar(&schema))
        .unwrap()
        .join()
        .unwrap();
    assert!(text.starts_with("char ::="));
}

#[test]
fn a_long_ref_chain_is_an_error_instead_of_a_stack_overflow() {
    let n = 100_000;
    let defs: Vec<String> = (0..n)
        .map(|i| format!(r##""a{i}": {{"$ref": "#/$defs/a{}"}}"##, i + 1))
        .chain([format!(r#""a{n}": {{"type": "string"}}"#)])
        .collect();
    let schema = format!(
        r##"{{"$ref": "#/$defs/a0", "$defs": {{{}}}}}"##,
        defs.join(",")
    );
    let result = std::thread::Builder::new()
        .stack_size(2 << 20)
        .spawn(move || json_schema_to_grammar(&schema).err())
        .unwrap()
        .join()
        .unwrap();
    assert!(result.unwrap().ends_with("schema nesting too deep"));
}

#[test]
fn self_referencing_refs_inside_all_of_terminate() {
    let schema = r##"{"allOf": [{"$ref": "#/$defs/d"}], "$defs": {"d": {"$ref": "#/$defs/d"}}}"##;
    let error = json_schema_to_grammar(schema).err().unwrap();
    assert!(error.ends_with("schema nesting too deep"));
}

#[test]
fn a_multibyte_character_before_a_quantifier_stays_whole() {
    let text = grammar(r#"{"type": "string", "pattern": "^é+x$"}"#);
    assert!(text.contains(r#"root ::= "\"" ("é"+ "x") "\"""#), "{text}");
}
