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

#[derive(Clone, Copy)]
enum Target {
    Model,
    Mmproj,
    Draft,
}

enum Done {
    Nothing,
    SetPath(Target, String),
    Finalize(HfFile),
    FinalizeInto(HfFile, Target),
    FinalizeDraftIfPathEmpty(HfFile),
    FinalizeDraftIfModelEmpty(HfFile),
    Preset(HfFile),
}

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

fn target(params: &mut Params, target: Target) -> &mut ModelParams {
    match target {
        Target::Model => &mut params.model,
        Target::Mmproj => &mut params.mmproj,
        Target::Draft => &mut params.speculative.draft.mparams,
    }
}

fn spec_types_is_default(params: &Params) -> bool {
    params.speculative.types == [SPEC_NONE]
}

fn hf_root() -> Result<PathBuf, Failure> {
    cache::cache_dir().ok_or_else(|| Failure::runtime(NO_HOME))
}

fn default_local_path(url: &str) -> Result<String, Failure> {
    let f = url.split('#').next().unwrap_or_default();
    let f = f.split('?').next().unwrap_or_default();
    let name = f.rsplit('/').next().unwrap_or_default();
    let dir = cache::llama_cache_dir().ok_or_else(|| Failure::runtime(NO_HOME))?;
    let mut dir = dir.to_string_lossy().into_owned();
    if !dir.ends_with('/') {
        dir.push('/');
    }
    std::fs::create_dir_all(&dir)
        .map_err(|_| Failure::runtime(format!("failed to create cache directory: {dir}")))?;
    Ok(dir + name)
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
        if spec_types_is_default(params) {
            opts_spec.wanted.mtp = true;
            opts_spec.wanted.dflash = true;
            opts_spec.wanted.eagle3 = true;
            opts_spec.wanted.dspark = true;
        }
        handler.plan_spec = hf_plan(draft, &opts_spec)?;
    }
    Ok(handler)
}

fn url_tasks(model: &ModelParams) -> Result<Vec<Task>, Failure> {
    let url = text(&model.url);
    let path = text(&model.path);
    let parts = gguf::all_parts(&url);

    if parts.len() == 1 {
        let local = if path.is_empty() {
            default_local_path(&parts[0])?
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
            let mut local = default_local_path(&part)?;
            if !base_dir.is_empty() {
                let name = match local.rfind('/') {
                    Some(pos) => &local[pos + 1..],
                    None => &local,
                };
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

fn finish(params: &mut Params, done: &Done) {
    match done {
        Done::Nothing => {}
        Done::SetPath(to, path) => target(params, *to).path = bytes(path.clone()),
        Done::Finalize(file) => {
            hub::finalize(file);
        }
        Done::FinalizeInto(file, to) => target(params, *to).path = bytes(hub::finalize(file)),
        Done::FinalizeDraftIfPathEmpty(file) => {
            let path = hub::finalize(file);
            let draft = &mut params.speculative.draft.mparams;
            if draft.path.is_empty() {
                draft.path = bytes(path);
            }
        }
        Done::FinalizeDraftIfModelEmpty(file) => {
            let empty = model_is_empty(&params.speculative.draft.mparams);
            let path = hub::finalize(file);
            if empty {
                params.speculative.draft.mparams.path = bytes(path);
            }
        }
        Done::Preset(file) => {
            params.models_preset_hf = params.model.hf_repo.clone();
            params.models_preset = bytes(hub::finalize(file));
            params.model = empty_model();
        }
    }
}

pub fn apply(
    handler: &mut Handler,
    params: &mut Params,
    callback: Option<&dyn Callback>,
    spec_types_from_gguf: &dyn Fn(&str) -> Vec<i32>,
) -> Result<(), Failure> {
    let mut tasks = Vec::new();

    for to in [Target::Model, Target::Mmproj, Target::Draft] {
        let model = target(params, to);
        if !model.url.is_empty() && model.path.is_empty() {
            model.path = bytes(default_local_path(&text(&model.url))?);
        }
    }

    if !params.model.docker_repo.is_empty() {
        let cache_dir = cache::llama_cache_dir().ok_or_else(|| Failure::runtime(NO_HOME))?;
        let url = docker::resolve_model(&text(&params.model.docker_repo), &cache_dir)
            .map_err(|e| Failure::runtime(e.to_string()))?;
        let url = url.to_string_lossy().into_owned();
        params.model.path = bytes(default_local_path(&url)?);
        params.model.url = bytes(url);
    }

    if !params.model.url.is_empty() {
        let mut url_tasks = url_tasks(&params.model)?;
        if let Some(first) = url_tasks.first_mut() {
            first.done = Done::SetPath(Target::Model, first.local_path.clone());
        }
        tasks.extend(url_tasks);
    }
    if !params.mmproj.url.is_empty() {
        tasks.push(Task::url(
            text(&params.mmproj.url),
            text(&params.mmproj.path),
            Done::Nothing,
        ));
    }
    let mut had_spec_url = false;
    let draft = &params.speculative.draft.mparams;
    if !draft.url.is_empty() {
        tasks.push(Task::url(
            text(&draft.url),
            text(&draft.path),
            Done::Nothing,
        ));
        had_spec_url = true;
    }

    let plan = &handler.plan;
    let plan_spec = &mut handler.plan_spec;

    if !params.speculative.draft.mparams.hf_file.is_empty() {
        plan_spec.mtp = None;
        plan_spec.dflash = None;
        plan_spec.eagle3 = None;
        plan_spec.dspark = None;
    }

    if spec_types_is_default(params) {
        if has(&plan_spec.mtp) {
            params.speculative.types = vec![SPEC_DRAFT_MTP];
            plan_spec.dspark = None;
            plan_spec.dflash = None;
            plan_spec.eagle3 = None;
        } else if has(&plan_spec.dspark) {
            params.speculative.types = vec![SPEC_DRAFT_DSPARK];
            plan_spec.dflash = None;
            plan_spec.eagle3 = None;
        } else if has(&plan_spec.dflash) {
            params.speculative.types = vec![SPEC_DRAFT_DFLASH];
            plan_spec.eagle3 = None;
        } else if has(&plan_spec.eagle3) {
            params.speculative.types = vec![SPEC_DRAFT_EAGLE3];
        }
    }

    if spec_types_is_default(params) && !params.speculative.draft.mparams.path.is_empty() {
        let types = spec_types_from_gguf(&text(&params.speculative.draft.mparams.path));
        if !types.is_empty() {
            params.speculative.types = types;
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

    if !params.offline {
        let mut unique: Vec<&Task> = Vec::new();
        for task in &tasks {
            if !unique.iter().any(|t| t.local_path == task.local_path) {
                unique.push(task);
            }
        }
        for task in &unique {
            log::emit(
                Level::Debug,
                format!("download task: {} -> {}", task.url, task.local_path),
            );
        }
        run_tasks(&unique, &handler.opts, callback)?;
    }

    for task in &tasks {
        finish(params, &task.done);
    }
    Ok(())
}
