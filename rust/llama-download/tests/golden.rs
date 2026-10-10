use llama_download::gguf;
use llama_download::repo;
use llama_download::select::{self, Wanted};
use serde_json::{Value, json};

fn strings(value: &Value) -> Vec<String> {
    value.as_array().map_or_else(Vec::new, |items| {
        items
            .iter()
            .map(|item| item.as_str().unwrap().to_string())
            .collect()
    })
}

fn path(files: &[String], index: Option<usize>) -> Value {
    index.map_or(Value::Null, |i| json!(files[i]))
}

fn plan(args: &[&str], files: &[String]) -> Value {
    let flag = |i: usize| args[i] == "1";
    let wanted = Wanted {
        mmproj: flag(2),
        mtp: flag(3),
        dflash: flag(4),
        eagle3: flag(5),
        dspark: flag(6),
    };
    let (_, tag) = repo::split_repo_tag(args[0]).unwrap();
    let selection = select::select_hf_plan(files, args[1], tag, wanted).unwrap_or_default();
    json!({
        "preset": path(files, selection.preset),
        "primary": path(files, selection.primary),
        "model_files": selection.model_files.iter().map(|&i| files[i].clone()).collect::<Vec<_>>(),
        "mmproj": path(files, selection.mmproj),
        "mtp": path(files, selection.mtp),
        "dflash": path(files, selection.dflash),
        "eagle3": path(files, selection.eagle3),
        "dspark": path(files, selection.dspark),
    })
}

fn run(record: &Value) -> Value {
    let args: Vec<&str> = record["args"]
        .as_array()
        .unwrap()
        .iter()
        .map(|arg| arg.as_str().unwrap())
        .collect();
    let files = strings(&record["files"]);
    match record["op"].as_str().unwrap() {
        "split" => {
            let info = gguf::split_info(args[0]);
            json!([info.prefix, info.tag, info.index, info.count])
        }
        "bits" => json!(gguf::quant_bits(args[0])),
        "is_model" => json!(gguf::is_model(args[0])),
        "sibling" => path(
            &files,
            select::find_best_sibling(&files, args[0], args[1], args[2]),
        ),
        "mmproj" => path(
            &files,
            select::find_best_sibling(&files, args[0], "mmproj", ""),
        ),
        "mtp" => path(
            &files,
            select::find_best_sibling(&files, args[0], "mtp-", args[1]),
        ),
        "eagle3" => path(
            &files,
            select::find_best_sibling(&files, args[0], "eagle3-", args[1]),
        ),
        "dflash" => path(
            &files,
            select::find_best_sibling(&files, args[0], "dflash-", args[1]),
        ),
        "dspark" => path(
            &files,
            select::find_best_sibling(&files, args[0], "dspark-", args[1]),
        ),
        "best" => path(&files, select::find_best_model(&files, args[0])),
        "split_files" => {
            let file = files.iter().position(|p| p == args[0]).unwrap();
            let parts = gguf::split_files(&files, file);
            json!(parts.iter().map(|&i| files[i].clone()).collect::<Vec<_>>())
        }
        "all_parts" => json!(gguf::all_parts(args[0])),
        "repo" => match repo::split_repo_tag(args[0]) {
            Ok((name, tag)) => json!([name, tag]),
            Err(_) => json!({"error": true}),
        },
        "plan" => plan(&args, &files),
        op => panic!("unknown op {op}"),
    }
}

#[test]
fn matches_common_download_cpp() {
    let mismatches: Vec<String> = include_str!("golden/download.jsonl")
        .lines()
        .filter_map(|line| {
            let record: Value = serde_json::from_str(line).unwrap();
            let actual = run(&record);
            (actual != record["out"]).then(|| {
                format!(
                    "{} {}\n  got      {actual}\n  expected {}",
                    record["op"], record["args"], record["out"]
                )
            })
        })
        .collect();
    assert!(mismatches.is_empty(), "{}", mismatches.join("\n"));
}
