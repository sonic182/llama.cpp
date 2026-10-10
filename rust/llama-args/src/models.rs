use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::thread;

use llama_download::cache;
use llama_download::docker;
use llama_download::ffi::{INVALID_REPO, NO_HOME};
use llama_download::gguf;
use llama_download::hub::{self, HfFile, Plan};
use llama_download::log::{self, Level};
use llama_download::progress;
use llama_download::remote::{self, Callback, Error, Options};
use llama_download::select::Wanted;
use serde_bytes::ByteBuf;

use crate::params::{ModelParams, Params};

const SPEC_NONE: i32 = 0;
const SPEC_DRAFT_EAGLE3: i32 = 2;
const SPEC_DRAFT_MTP: i32 = 3;
const SPEC_DRAFT_DFLASH: i32 = 4;
const SPEC_DRAFT_DSPARK: i32 = 5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    InvalidArgument = 0,
    Runtime = 1,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Failure {
    pub kind: ErrorKind,
    pub message: String,
}

impl Failure {
    fn runtime(message: impl Into<String>) -> Self {
        Failure {
            kind: ErrorKind::Runtime,
            message: message.into(),
        }
    }

    fn invalid_repo() -> Self {
        Failure {
            kind: ErrorKind::InvalidArgument,
            message: INVALID_REPO.to_string(),
        }
    }
}

#[derive(Debug, Clone, Default)]
struct Opts {
    bearer_token: String,
    offline: bool,
    wanted: Wanted,
}

#[derive(Debug, Default)]
pub struct Handler {
    plan: Plan,
    plan_spec: Plan,
    opts: Opts,
}

impl Handler {
    pub fn is_preset_repo(&self) -> bool {
        self.plan.preset.as_ref().is_some_and(|f| !f.url.is_empty())
    }
}

#[derive(Debug, Clone, PartialEq)]
struct Sources {
    model: ModelParams,
    mmproj: ModelParams,
    draft: ModelParams,
    spec_types: Vec<i32>,
    models_preset: ByteBuf,
    models_preset_hf: ByteBuf,
}

impl Sources {
    fn from_params(params: &Params) -> Self {
        Sources {
            model: params.model.clone(),
            mmproj: params.mmproj.clone(),
            draft: params.speculative.draft.mparams.clone(),
            spec_types: params.speculative.types.clone(),
            models_preset: params.models_preset.clone(),
            models_preset_hf: params.models_preset_hf.clone(),
        }
    }

    fn store(self, params: &mut Params) {
        params.model = self.model;
        params.mmproj = self.mmproj;
        params.speculative.draft.mparams = self.draft;
        params.speculative.types = self.spec_types;
        params.models_preset = self.models_preset;
        params.models_preset_hf = self.models_preset_hf;
    }

    fn target(&mut self, target: Target) -> &mut ModelParams {
        match target {
            Target::Model => &mut self.model,
            Target::Mmproj => &mut self.mmproj,
            Target::Draft => &mut self.draft,
        }
    }

    fn spec_types_is_default(&self) -> bool {
        self.spec_types == [SPEC_NONE]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Target {
    Model,
    Mmproj,
    Draft,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Done {
    Nothing,
    SetPath(Target, String),
    Finalize(HfFile),
    FinalizeInto(HfFile, Target),
    FinalizeDraftIfPathEmpty(HfFile),
    FinalizeDraftIfModelEmpty(HfFile),
    Preset(HfFile),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Task {
    url: String,
    local_path: String,
    is_hf: bool,
    done: Done,
}

impl Task {
    fn url(url: String, local_path: String, done: Done) -> Self {
        Task {
            url,
            local_path,
            is_hf: false,
            done,
        }
    }

    fn hf(file: &HfFile, done: Done) -> Self {
        Task {
            url: file.url.clone(),
            local_path: file.local_path.clone(),
            is_hf: true,
            done,
        }
    }
}

type LocalPath<'a> = &'a dyn Fn(&str) -> Result<String, Failure>;

fn text(b: &ByteBuf) -> String {
    String::from_utf8_lossy(b).into_owned()
}

fn bytes(s: String) -> ByteBuf {
    ByteBuf::from(s.into_bytes())
}

fn has(file: &Option<HfFile>) -> bool {
    file.as_ref().is_some_and(|f| !f.local_path.is_empty())
}

fn model_is_empty(model: &ModelParams) -> bool {
    model.hf_repo.is_empty() && model.docker_repo.is_empty() && model.path.is_empty()
}

fn empty_model() -> ModelParams {
    ModelParams {
        path: ByteBuf::new(),
        url: ByteBuf::new(),
        hf_repo: ByteBuf::new(),
        hf_file: ByteBuf::new(),
        docker_repo: ByteBuf::new(),
    }
}

fn hf_root() -> Result<PathBuf, Failure> {
    cache::cache_dir().ok_or_else(|| Failure::runtime(NO_HOME))
}

fn llama_cache_dir() -> Result<PathBuf, Failure> {
    cache::llama_cache_dir().ok_or_else(|| Failure::runtime(NO_HOME))
}

fn file_name(url: &str) -> &str {
    let f = url.split('#').next().unwrap_or_default();
    let f = f.split('?').next().unwrap_or_default();
    f.rsplit('/').next().unwrap_or_default()
}

fn default_local_path(url: &str) -> Result<String, Failure> {
    let mut dir = llama_cache_dir()?.to_string_lossy().into_owned();
    if !dir.ends_with('/') {
        dir.push('/');
    }
    std::fs::create_dir_all(&dir)
        .map_err(|_| Failure::runtime(format!("failed to create cache directory: {dir}")))?;
    Ok(dir + file_name(url))
}

fn hf_plan(model: &ModelParams, opts: &Opts) -> Result<Plan, Failure> {
    hub::plan(
        &hf_root()?,
        &text(&model.hf_repo),
        &text(&model.hf_file),
        opts.offline,
        &opts.bearer_token,
        opts.wanted,
    )
    .map_err(|_| Failure::invalid_repo())
}

pub fn init(params: &Params, use_mmproj: bool) -> Result<Handler, Failure> {
    let has_type = |t: i32| params.speculative.types.contains(&t);
    let opts = Opts {
        bearer_token: text(&params.hf_token),
        offline: params.offline,
        wanted: Wanted {
            mmproj: use_mmproj
                && !params.no_mmproj
                && params.mmproj.path.is_empty()
                && params.mmproj.url.is_empty(),
            mtp: has_type(SPEC_DRAFT_MTP),
            eagle3: has_type(SPEC_DRAFT_EAGLE3),
            dflash: has_type(SPEC_DRAFT_DFLASH),
            dspark: has_type(SPEC_DRAFT_DSPARK),
        },
    };

    let mut handler = Handler {
        opts,
        ..Default::default()
    };
    if !params.model.hf_repo.is_empty() {
        handler.plan = hf_plan(&params.model, &handler.opts)?;
    }
    let draft = &params.speculative.draft.mparams;
    if !draft.hf_repo.is_empty() {
        let mut opts_spec = handler.opts.clone();
        if params.speculative.types == [SPEC_NONE] {
            opts_spec.wanted.mtp = true;
            opts_spec.wanted.dflash = true;
            opts_spec.wanted.eagle3 = true;
            opts_spec.wanted.dspark = true;
        }
        handler.plan_spec = hf_plan(draft, &opts_spec)?;
    }
    Ok(handler)
}

fn url_tasks(model: &ModelParams, local_path: LocalPath) -> Result<Vec<Task>, Failure> {
    let url = text(&model.url);
    let path = text(&model.path);
    let parts = gguf::all_parts(&url);

    if parts.len() == 1 {
        let local = if path.is_empty() {
            local_path(&parts[0])?
        } else {
            path
        };
        return Ok(vec![Task::url(parts[0].clone(), local, Done::Nothing)]);
    }

    let base_dir = if path.is_empty() {
        String::new()
    } else {
        match path.rfind('/') {
            Some(pos) => path[..pos].to_string(),
            None => ".".to_string(),
        }
    };
    parts
        .into_iter()
        .map(|part| {
            let mut local = local_path(&part)?;
            if !base_dir.is_empty() {
                let name = local.rsplit('/').next().unwrap_or_default();
                local = format!("{base_dir}/{name}");
            }
            Ok(Task::url(part, local, Done::Nothing))
        })
        .collect()
}

fn add_model_tasks(tasks: &mut Vec<Task>, files: &[HfFile], primary: &Option<HfFile>, to: Target) {
    let primary = primary.as_ref().map_or("", |f| f.path.as_str());
    for file in files {
        let done = if file.path == primary {
            Done::FinalizeInto(file.clone(), to)
        } else {
            Done::Finalize(file.clone())
        };
        tasks.push(Task::hf(file, done));
    }
}

fn plan_tasks(
    plan: &Plan,
    plan_spec: &mut Plan,
    src: &mut Sources,
    local_path: LocalPath,
    spec_types_from_gguf: &dyn Fn(&str) -> Vec<i32>,
) -> Result<Vec<Task>, Failure> {
    let mut tasks = Vec::new();

    for to in [Target::Model, Target::Mmproj, Target::Draft] {
        let model = src.target(to);
        if !model.url.is_empty() && model.path.is_empty() {
            model.path = bytes(local_path(&text(&model.url))?);
        }
    }

    if !src.model.url.is_empty() {
        let mut url_tasks = url_tasks(&src.model, local_path)?;
        if let Some(first) = url_tasks.first_mut() {
            first.done = Done::SetPath(Target::Model, first.local_path.clone());
        }
        tasks.extend(url_tasks);
    }
    if !src.mmproj.url.is_empty() {
        tasks.push(Task::url(
            text(&src.mmproj.url),
            text(&src.mmproj.path),
            Done::Nothing,
        ));
    }
    let mut had_spec_url = false;
    if !src.draft.url.is_empty() {
        tasks.push(Task::url(
            text(&src.draft.url),
            text(&src.draft.path),
            Done::Nothing,
        ));
        had_spec_url = true;
    }

    if !src.draft.hf_file.is_empty() {
        plan_spec.mtp = None;
        plan_spec.dflash = None;
        plan_spec.eagle3 = None;
        plan_spec.dspark = None;
    }

    if src.spec_types_is_default() {
        if has(&plan_spec.mtp) {
            src.spec_types = vec![SPEC_DRAFT_MTP];
            plan_spec.dspark = None;
            plan_spec.dflash = None;
            plan_spec.eagle3 = None;
        } else if has(&plan_spec.dspark) {
            src.spec_types = vec![SPEC_DRAFT_DSPARK];
            plan_spec.dflash = None;
            plan_spec.eagle3 = None;
        } else if has(&plan_spec.dflash) {
            src.spec_types = vec![SPEC_DRAFT_DFLASH];
            plan_spec.eagle3 = None;
        } else if has(&plan_spec.eagle3) {
            src.spec_types = vec![SPEC_DRAFT_EAGLE3];
        }
    }

    if src.spec_types_is_default() && !src.draft.path.is_empty() {
        let types = spec_types_from_gguf(&text(&src.draft.path));
        if !types.is_empty() {
            src.spec_types = types;
        }
    }

    let spec_sidecars = [
        &plan_spec.mtp,
        &plan_spec.dflash,
        &plan_spec.eagle3,
        &plan_spec.dspark,
    ];
    let spec_sidecar_found = spec_sidecars.iter().any(|f| has(f));
    if !had_spec_url {
        for file in spec_sidecars.into_iter().flatten() {
            if !file.local_path.is_empty() {
                tasks.push(Task::hf(file, Done::FinalizeDraftIfPathEmpty(file.clone())));
            }
        }
    }
    if spec_sidecar_found {
        had_spec_url = true;
    }

    if !plan_spec.model_files.is_empty() && !had_spec_url && !spec_sidecar_found {
        add_model_tasks(
            &mut tasks,
            &plan_spec.model_files,
            &plan_spec.primary,
            Target::Draft,
        );
        had_spec_url = true;
    }

    if !plan.model_files.is_empty() {
        add_model_tasks(&mut tasks, &plan.model_files, &plan.primary, Target::Model);
    }
    if let Some(file) = plan.mmproj.as_ref().filter(|f| !f.local_path.is_empty()) {
        tasks.push(Task::hf(
            file,
            Done::FinalizeInto(file.clone(), Target::Mmproj),
        ));
    }
    if !had_spec_url {
        for file in [&plan.mtp, &plan.dflash, &plan.eagle3, &plan.dspark]
            .into_iter()
            .flatten()
        {
            if !file.local_path.is_empty() {
                tasks.push(Task::hf(
                    file,
                    Done::FinalizeDraftIfModelEmpty(file.clone()),
                ));
            }
        }
    }
    if let Some(file) = plan.preset.as_ref().filter(|f| !f.local_path.is_empty()) {
        tasks.push(Task::hf(file, Done::Preset(file.clone())));
    }

    Ok(tasks)
}

fn unique_tasks(tasks: &[Task]) -> Vec<&Task> {
    let mut unique: Vec<&Task> = Vec::new();
    for task in tasks {
        if !unique.iter().any(|t| t.local_path == task.local_path) {
            unique.push(task);
        }
    }
    unique
}

fn finish(src: &mut Sources, done: &Done, finalize: &dyn Fn(&HfFile) -> String) {
    match done {
        Done::Nothing => {}
        Done::SetPath(to, path) => src.target(*to).path = bytes(path.clone()),
        Done::Finalize(file) => {
            finalize(file);
        }
        Done::FinalizeInto(file, to) => src.target(*to).path = bytes(finalize(file)),
        Done::FinalizeDraftIfPathEmpty(file) => {
            let path = finalize(file);
            if src.draft.path.is_empty() {
                src.draft.path = bytes(path);
            }
        }
        Done::FinalizeDraftIfModelEmpty(file) => {
            let empty = model_is_empty(&src.draft);
            let path = finalize(file);
            if empty {
                src.draft.path = bytes(path);
            }
        }
        Done::Preset(file) => {
            src.models_preset_hf = src.model.hf_repo.clone();
            src.models_preset = bytes(finalize(file));
            src.model = empty_model();
        }
    }
}

fn resolve_docker(src: &mut Sources, local_path: LocalPath) -> Result<(), Failure> {
    if src.model.docker_repo.is_empty() {
        return Ok(());
    }
    let url = docker::resolve_model(&text(&src.model.docker_repo), &llama_cache_dir()?)
        .map_err(|e| Failure::runtime(e.to_string()))?;
    let url = url.to_string_lossy().into_owned();
    src.model.path = bytes(local_path(&url)?);
    src.model.url = bytes(url);
    Ok(())
}

fn download(task: &Task, opts: &Opts, callback: Option<&dyn Callback>) -> i32 {
    let bar = progress::stdout_bar();
    let opts = Options {
        headers: Vec::new(),
        bearer_token: (!opts.bearer_token.is_empty()).then(|| opts.bearer_token.clone()),
        offline: opts.offline,
        callback: Some(callback.unwrap_or(&bar)),
    };
    let result = catch_unwind(AssertUnwindSafe(|| {
        remote::download_file(&task.url, Path::new(&task.local_path), &opts, task.is_hf)
    }));
    match result {
        Ok(Ok(status)) | Ok(Err(Error::Status(status))) => i32::from(status),
        _ => -1,
    }
}

fn run_tasks(tasks: &[&Task], opts: &Opts, callback: Option<&dyn Callback>) -> Result<(), Failure> {
    for task in tasks {
        log::emit(
            Level::Debug,
            format!("download task: {} -> {}", task.url, task.local_path),
        );
    }
    let statuses: Vec<i32> = thread::scope(|scope| {
        let handles: Vec<_> = tasks
            .iter()
            .map(|task| scope.spawn(move || download(task, opts, callback)))
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().unwrap_or(-1))
            .collect()
    });
    for (task, status) in tasks.iter().zip(statuses) {
        if !(200..400).contains(&status) {
            return Err(Failure::runtime(format!(
                "Download '{}' failed with status code: {status}",
                task.url
            )));
        }
    }
    Ok(())
}

pub fn apply(
    handler: &mut Handler,
    params: &mut Params,
    callback: Option<&dyn Callback>,
    spec_types_from_gguf: &dyn Fn(&str) -> Vec<i32>,
) -> Result<(), Failure> {
    let mut src = Sources::from_params(params);
    let local_path: LocalPath = &default_local_path;
    resolve_docker(&mut src, local_path)?;

    let tasks = plan_tasks(
        &handler.plan,
        &mut handler.plan_spec,
        &mut src,
        local_path,
        spec_types_from_gguf,
    )?;

    if !params.offline {
        run_tasks(&unique_tasks(&tasks), &handler.opts, callback)?;
    }

    for task in &tasks {
        finish(&mut src, &task.done, &hub::finalize);
    }
    src.store(params);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(hf_repo: &str, hf_file: &str, url: &str, path: &str) -> ModelParams {
        ModelParams {
            path: bytes(path.into()),
            url: bytes(url.into()),
            hf_repo: bytes(hf_repo.into()),
            hf_file: bytes(hf_file.into()),
            docker_repo: ByteBuf::new(),
        }
    }

    fn sources(main: ModelParams, draft: ModelParams) -> Sources {
        Sources {
            model: main,
            mmproj: empty_model(),
            draft,
            spec_types: vec![SPEC_NONE],
            models_preset: ByteBuf::new(),
            models_preset_hf: ByteBuf::new(),
        }
    }

    fn hf(repo: &str, path: &str) -> HfFile {
        HfFile {
            path: path.into(),
            url: format!("https://hf/{repo}/{path}"),
            local_path: format!("blobs/{repo}/{path}"),
            final_path: format!("snapshots/{repo}/{path}"),
            ..Default::default()
        }
    }

    fn local_path(url: &str) -> Result<String, Failure> {
        Ok(format!("cache/{}", file_name(url)))
    }

    fn no_gguf(path: &str) -> Vec<i32> {
        panic!("unexpected gguf read of {path}")
    }

    fn finalize(file: &HfFile) -> String {
        file.final_path.clone()
    }

    fn run(plan: &Plan, plan_spec: &mut Plan, src: &mut Sources) -> Vec<Task> {
        let tasks = plan_tasks(plan, plan_spec, src, &local_path, &no_gguf).unwrap();
        for task in &tasks {
            finish(src, &task.done, &finalize);
        }
        tasks
    }

    fn path(m: &ModelParams) -> String {
        text(&m.path)
    }

    #[test]
    fn draft_repo_sidecar_sets_the_type_and_wins_over_the_main_sidecar() {
        let plan = Plan {
            primary: Some(hf("main", "m-Q8_0.gguf")),
            model_files: vec![hf("main", "m-Q8_0.gguf")],
            mtp: Some(hf("main", "mtp-m.gguf")),
            ..Default::default()
        };
        let mut plan_spec = Plan {
            mtp: Some(hf("draft", "mtp-d.gguf")),
            dflash: Some(hf("draft", "dflash-d.gguf")),
            ..Default::default()
        };
        let mut src = sources(model("main", "", "", ""), model("draft", "", "", ""));

        let tasks = run(&plan, &mut plan_spec, &mut src);

        assert_eq!(src.spec_types, [SPEC_DRAFT_MTP]);
        assert_eq!(plan_spec.dflash, None);
        assert_eq!(path(&src.model), "snapshots/main/m-Q8_0.gguf");
        assert_eq!(path(&src.draft), "snapshots/draft/mtp-d.gguf");
        let urls: Vec<&str> = tasks.iter().map(|t| t.url.as_str()).collect();
        assert_eq!(
            urls,
            ["https://hf/draft/mtp-d.gguf", "https://hf/main/m-Q8_0.gguf"]
        );
    }

    #[test]
    fn explicit_draft_file_skips_sidecars_and_downloads_the_draft_model() {
        let plan = Plan {
            mtp: Some(hf("main", "mtp-m.gguf")),
            ..Default::default()
        };
        let mut plan_spec = Plan {
            primary: Some(hf("draft", "d-00001-of-00002.gguf")),
            model_files: vec![
                hf("draft", "d-00001-of-00002.gguf"),
                hf("draft", "d-00002-of-00002.gguf"),
            ],
            mtp: Some(hf("draft", "mtp-d.gguf")),
            ..Default::default()
        };
        let mut src = sources(
            empty_model(),
            model("draft", "d-00001-of-00002.gguf", "", ""),
        );

        let tasks = run(&plan, &mut plan_spec, &mut src);

        assert_eq!(src.spec_types, [SPEC_NONE]);
        assert_eq!(path(&src.draft), "snapshots/draft/d-00001-of-00002.gguf");
        assert!(tasks.iter().all(|t| !t.url.contains("mtp")));
        assert_eq!(tasks.len(), 2);
    }

    #[test]
    fn main_sidecar_is_the_draft_only_when_no_draft_was_given() {
        let plan = Plan {
            dflash: Some(hf("main", "dflash-m.gguf")),
            ..Default::default()
        };

        let mut src = sources(model("main", "", "", ""), empty_model());
        run(&plan, &mut Plan::default(), &mut src);
        assert_eq!(path(&src.draft), "snapshots/main/dflash-m.gguf");

        let mut src = sources(model("main", "", "", ""), model("", "", "", "own.gguf"));
        let gguf = |p: &str| {
            assert_eq!(p, "own.gguf");
            vec![SPEC_DRAFT_DFLASH]
        };
        let tasks = plan_tasks(&plan, &mut Plan::default(), &mut src, &local_path, &gguf).unwrap();
        for task in &tasks {
            finish(&mut src, &task.done, &finalize);
        }
        assert_eq!(src.spec_types, [SPEC_DRAFT_DFLASH]);
        assert_eq!(path(&src.draft), "own.gguf");
    }

    #[test]
    fn preset_repo_clears_the_model_for_router_mode() {
        let plan = Plan {
            preset: Some(hf("presets", "preset.ini")),
            ..Default::default()
        };
        let mut src = sources(model("presets", "", "", ""), empty_model());

        run(&plan, &mut Plan::default(), &mut src);

        assert_eq!(text(&src.models_preset), "snapshots/presets/preset.ini");
        assert_eq!(text(&src.models_preset_hf), "presets");
        assert_eq!(src.model, empty_model());
    }

    #[test]
    fn split_url_parts_go_next_to_the_model_path() {
        let mut src = sources(
            model("", "", "https://x/m-00001-of-00002.gguf", "dir/m.gguf"),
            empty_model(),
        );

        let tasks = run(&Plan::default(), &mut Plan::default(), &mut src);

        let locals: Vec<&str> = tasks.iter().map(|t| t.local_path.as_str()).collect();
        assert_eq!(
            locals,
            ["dir/m-00001-of-00002.gguf", "dir/m-00002-of-00002.gguf"]
        );
        assert_eq!(path(&src.model), "dir/m-00001-of-00002.gguf");
    }

    #[test]
    fn unique_tasks_keep_the_first_task_per_local_path() {
        let task = |url: &str, local: &str| Task::url(url.into(), local.into(), Done::Nothing);
        let tasks = [task("a", "x"), task("b", "y"), task("c", "x")];

        let urls: Vec<&str> = unique_tasks(&tasks)
            .iter()
            .map(|t| t.url.as_str())
            .collect();

        assert_eq!(urls, ["a", "b"]);
    }
}
