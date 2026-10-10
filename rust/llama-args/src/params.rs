use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use serde_bytes::ByteBuf;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Params {
    pub n_predict: i32,
    pub n_ctx: i32,
    pub n_batch: i32,
    pub n_ubatch: i32,
    pub n_keep: i32,
    pub n_chunks: i32,
    pub n_parallel: i32,
    pub n_sequences: i32,
    pub n_outputs_max: i32,
    pub n_outputs_max_per_seq: i32,
    pub grp_attn_n: i32,
    pub grp_attn_w: i32,
    pub n_print: i32,
    pub rope_freq_base: f32,
    pub rope_freq_scale: f32,
    pub yarn_ext_factor: f32,
    pub yarn_attn_factor: f32,
    pub yarn_beta_fast: f32,
    pub yarn_beta_slow: f32,
    pub yarn_orig_ctx: i32,
    pub devices: Vec<Option<ByteBuf>>,
    pub n_gpu_layers: i32,
    pub main_gpu: i32,
    pub tensor_split: Vec<f32>,
    pub fit_params: bool,
    pub fit_params_print: bool,
    pub fit_params_min_ctx: i32,
    pub fit_params_target: Vec<u64>,
    pub split_mode: i32,
    pub load_mode: i32,
    pub lazy_mode: i32,
    pub cpuparams: CpuParams,
    pub cpuparams_batch: CpuParams,
    pub numa: i32,
    pub rope_scaling_type: i32,
    pub pooling_type: i32,
    pub attention_type: i32,
    pub flash_attn_type: i32,
    pub sampling: SamplingParams,
    pub speculative: SpeculativeParams,
    pub diffusion: DiffusionParams,
    pub model: ModelParams,
    pub model_alias: BTreeSet<ByteBuf>,
    pub model_tags: BTreeSet<ByteBuf>,
    pub hf_token: ByteBuf,
    pub prompt: ByteBuf,
    pub system_prompt: ByteBuf,
    pub prompt_file: ByteBuf,
    pub path_prompt_cache: ByteBuf,
    pub input_prefix: ByteBuf,
    pub input_suffix: ByteBuf,
    pub logits_file: ByteBuf,
    pub path_prompts_log_dir: ByteBuf,
    pub logits_output_dir: ByteBuf,
    pub save_logits: bool,
    pub tensor_filter: Vec<ByteBuf>,
    pub in_files: Vec<ByteBuf>,
    pub antiprompt: Vec<ByteBuf>,
    pub kv_overrides: Vec<KvOverride>,
    pub tensor_buft_overrides: Vec<TensorBuftOverride>,
    pub lora_init_without_apply: bool,
    pub lora_adapters: Vec<LoraAdapter>,
    pub control_vectors: Vec<ControlVector>,
    pub verbosity: i32,
    pub control_vector_layer_start: i32,
    pub control_vector_layer_end: i32,
    pub offline: bool,
    pub ppl_stride: i32,
    pub ppl_output_type: i32,
    pub hellaswag: bool,
    pub hellaswag_tasks: u64,
    pub winogrande: bool,
    pub winogrande_tasks: u64,
    pub multiple_choice: bool,
    pub multiple_choice_tasks: u64,
    pub kl_divergence: bool,
    pub check: bool,
    pub usage: bool,
    pub completion: bool,
    pub use_color: bool,
    pub special: bool,
    pub interactive: bool,
    pub interactive_first: bool,
    pub prompt_cache_all: bool,
    pub prompt_cache_ro: bool,
    pub escape: bool,
    pub multiline_input: bool,
    pub simple_io: bool,
    pub cont_batching: bool,
    pub no_perf: bool,
    pub show_timings: bool,
    pub ctx_shift: bool,
    pub swa_full: bool,
    pub kv_unified: bool,
    pub input_prefix_bos: bool,
    pub verbose_prompt: bool,
    pub display_prompt: bool,
    pub no_kv_offload: bool,
    pub warmup: bool,
    pub check_tensors: bool,
    pub no_op_offload: bool,
    pub no_extra_bufts: bool,
    pub no_host: bool,
    pub single_turn: bool,
    pub cache_type_k: i32,
    pub cache_type_v: i32,
    pub conversation_mode: i32,
    pub mmproj: ModelParams,
    pub mmproj_use_gpu: bool,
    pub mmproj_device: Option<ByteBuf>,
    pub no_mmproj: bool,
    pub image: Vec<ByteBuf>,
    pub image_min_tokens: i32,
    pub image_max_tokens: i32,
    pub mtmd_batch_max_tokens: i32,
    pub video_fps: f32,
    pub video_timestamp_interval_ms: i64,
    pub video_ffmpeg_bin_dir: ByteBuf,
    pub lr: LrOpt,
    pub optimizer: i32,
    pub val_split: f32,
    pub embedding: bool,
    pub embd_normalize: i32,
    pub embd_out: ByteBuf,
    pub embd_sep: ByteBuf,
    pub cls_sep: ByteBuf,
    pub port: i32,
    pub reuse_port: bool,
    pub timeout_read: i32,
    pub timeout_write: i32,
    pub sse_ping_interval: i32,
    pub n_threads_http: i32,
    pub http_workers: i32,
    pub n_cache_reuse: i32,
    pub cache_prompt: bool,
    pub cache_idle_slots: bool,
    pub n_ctx_checkpoints: i32,
    pub kv_unified_per_slot: i32,
    pub checkpoint_min_step: i32,
    pub cache_ram_mib: i32,
    pub public_path: ByteBuf,
    pub api_prefix: ByteBuf,
    pub chat_template: ByteBuf,
    pub hostnames: Vec<ByteBuf>,
    pub use_jinja: bool,
    pub cors_origins: ByteBuf,
    pub cors_methods: ByteBuf,
    pub cors_headers: ByteBuf,
    pub cors_credentials: bool,
    pub cors_origins_explicit: bool,
    pub enable_chat_template: bool,
    pub force_pure_content_parser: bool,
    pub reasoning_format: i32,
    pub enable_reasoning: i32,
    pub prefill_assistant: bool,
    pub sleep_idle_seconds: i32,
    pub api_keys: Vec<ByteBuf>,
    pub ssl_file_key: ByteBuf,
    pub ssl_file_cert: ByteBuf,
    pub default_template_kwargs: Vec<(ByteBuf, ByteBuf)>,
    pub preserve_reasoning_specified: bool,
    pub server_base: ByteBuf,
    pub ui: bool,
    pub ui_mcp_proxy: bool,
    pub ui_config_json: ByteBuf,
    pub endpoint_slots: bool,
    pub endpoint_props: bool,
    pub endpoint_metrics: bool,
    pub server_tools: Vec<ByteBuf>,
    pub server_tools_runtime: ByteBuf,
    pub mcp_servers_config: ByteBuf,
    pub mcp_servers_json: ByteBuf,
    pub models_dir: ByteBuf,
    pub models_preset: ByteBuf,
    pub models_max: i32,
    pub models_autoload: bool,
    pub models_preset_hf: ByteBuf,
    pub log_json: bool,
    pub slot_save_path: ByteBuf,
    pub media_path: ByteBuf,
    pub slot_prompt_similarity: f32,
    pub is_pp_shared: bool,
    pub is_tg_separate: bool,
    pub n_pp: Vec<i32>,
    pub n_tg: Vec<i32>,
    pub n_pl: Vec<i32>,
    pub context_files: Vec<ByteBuf>,
    pub chunk_size: i32,
    pub chunk_separator: ByteBuf,
    pub n_junk: i32,
    pub i_pos: i32,
    pub n_out_freq: i32,
    pub n_save_freq: i32,
    pub i_chunk: i32,
    pub imat_dat: i8,
    pub process_output: bool,
    pub compute_ppl: bool,
    pub show_statistics: bool,
    pub parse_special: bool,
    pub n_pca_batch: i32,
    pub n_pca_iterations: i32,
    pub cvector_dimre_method: i32,
    pub cvector_positive_file: ByteBuf,
    pub cvector_negative_file: ByteBuf,
    pub spm_infill: bool,
    pub batched_bench_output_jsonl: bool,
    pub tokenize_ids: bool,
    pub tokenize_stdin: bool,
    pub tokenize_no_bos: bool,
    pub tokenize_show_count: bool,
    pub out_file: ByteBuf,
    pub no_alloc: bool,
    pub tts_lang: ByteBuf,
    pub tts_speaker_file: ByteBuf,
    pub is_gen_docs: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CpuParams {
    pub n_threads: i32,
    pub cpumask: Vec<bool>,
    pub mask_valid: bool,
    pub priority: i32,
    pub strict_cpu: bool,
    pub poll: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SamplingParams {
    pub seed: u32,
    pub n_prev: i32,
    pub n_probs: i32,
    pub min_keep: i32,
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
    pub xtc_probability: f32,
    pub xtc_threshold: f32,
    pub typ_p: f32,
    pub temp: f32,
    pub dynatemp_range: f32,
    pub dynatemp_exponent: f32,
    pub penalty_last_n: i32,
    pub penalty_repeat: f32,
    pub penalty_freq: f32,
    pub penalty_present: f32,
    pub dry_multiplier: f32,
    pub dry_base: f32,
    pub dry_allowed_length: i32,
    pub dry_penalty_last_n: i32,
    pub adaptive_target: f32,
    pub adaptive_decay: f32,
    pub mirostat: i32,
    pub top_n_sigma: f32,
    pub mirostat_tau: f32,
    pub mirostat_eta: f32,
    pub ignore_eos: bool,
    pub no_perf: bool,
    pub timing_per_token: bool,
    pub user_sampling_config: u64,
    pub dry_sequence_breakers: Vec<ByteBuf>,
    pub samplers: Vec<i32>,
    pub grammar: Grammar,
    pub grammar_lazy: bool,
    pub grammar_triggers: Vec<GrammarTrigger>,
    pub preserved_tokens: BTreeSet<i32>,
    pub logit_bias: Vec<LogitBias>,
    pub logit_bias_eog: Vec<LogitBias>,
    pub generation_prompt: ByteBuf,
    pub reasoning_budget_tokens: i32,
    pub reasoning_budget_start: Vec<i32>,
    pub reasoning_budget_end: Vec<Vec<i32>>,
    pub reasoning_budget_forced: Vec<i32>,
    pub reasoning_budget_message: ByteBuf,
    pub reasoning_control: bool,
    pub backend_sampling: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpeculativeParams {
    pub types: Vec<i32>,
    pub synth_len: f64,
    pub synth_rates: Vec<f64>,
    pub draft: DraftParams,
    pub ngram_mod: NgramModParams,
    pub ngram_simple: NgramMapParams,
    pub ngram_map_k: NgramMapParams,
    pub ngram_map_k4v: NgramMapParams,
    pub ngram_cache: NgramCacheParams,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiffusionParams {
    pub steps: i32,
    pub visual_mode: bool,
    pub eps: f32,
    pub block_length: i32,
    pub algorithm: i32,
    pub alg_temp: f32,
    pub cfg_scale: f32,
    pub add_gumbel_noise: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelParams {
    pub path: ByteBuf,
    pub url: ByteBuf,
    pub hf_repo: ByteBuf,
    pub hf_file: ByteBuf,
    pub docker_repo: ByteBuf,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LoraAdapter {
    pub path: ByteBuf,
    pub scale: f32,
    pub task_name: ByteBuf,
    pub prompt_prefix: ByteBuf,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ControlVector {
    pub strength: f32,
    pub fname: ByteBuf,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LrOpt {
    pub lr0: f32,
    pub lr_min: f32,
    pub decay_epochs: f32,
    pub scale_epoch: f32,
    pub wd: f32,
    pub epochs: u32,
    pub epoch: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Grammar {
    pub r#type: i32,
    pub grammar: ByteBuf,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GrammarTrigger {
    pub r#type: i32,
    pub value: ByteBuf,
    pub token: i32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DraftParams {
    pub n_max: i32,
    pub n_min: i32,
    pub p_split: f32,
    pub p_min: f32,
    pub backend_sampling: bool,
    pub mparams: ModelParams,
    pub n_gpu_layers: i32,
    pub cache_type_k: i32,
    pub cache_type_v: i32,
    pub cpuparams: CpuParams,
    pub cpuparams_batch: CpuParams,
    pub devices: Vec<Option<ByteBuf>>,
    pub tensor_buft_overrides: Vec<TensorBuftOverride>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NgramModParams {
    pub n_match: i32,
    pub n_max: i32,
    pub n_min: i32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NgramMapParams {
    pub size_n: u16,
    pub size_m: u16,
    pub min_hits: u16,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NgramCacheParams {
    pub lookup_cache_static: ByteBuf,
    pub lookup_cache_dynamic: ByteBuf,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KvOverride {
    pub key: ByteBuf,
    pub tag: i32,
    pub value: KvValue,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum KvValue {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(ByteBuf),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorBuftOverride {
    pub pattern: Option<ByteBuf>,
    pub buft: Option<ByteBuf>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LogitBias {
    pub token: i32,
    pub bias: f32,
}
