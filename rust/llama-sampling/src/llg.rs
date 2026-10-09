use std::ffi::{CStr, CString, c_char, c_void};
use std::ptr;
use std::slice;
use std::sync::Mutex;

use llama::sys::{
    LLAMA_TOKEN_NULL, ggml_abort, llama_detokenize, llama_sampler, llama_sampler_i,
    llama_sampler_init, llama_token, llama_token_data_array, llama_tokenize, llama_vocab,
    llama_vocab_eos, llama_vocab_eot, llama_vocab_n_tokens,
};
use llguidance::ffi::{
    LlgConstraintInit, LlgMatcher, LlgTokenizer, LlgTokenizerInit, llg_clone_matcher,
    llg_clone_tokenizer, llg_constraint_init_set_defaults, llg_free_matcher, llg_free_tokenizer,
    llg_matcher_compute_mask, llg_matcher_consume_token, llg_matcher_get_error,
    llg_matcher_get_mask, llg_matcher_get_mask_byte_size, llg_matcher_reset, llg_new_matcher,
    llg_new_tokenizer,
};

use crate::log;

const FUNC: &CStr = c"llama_sampler_init_llg";

struct Ctx {
    vocab: *const llama_vocab,
    grammar_kind: CString,
    grammar_data: CString,
    tokenizer: *mut LlgTokenizer,
    grammar: *mut LlgMatcher,
}

static IFACE: llama_sampler_i = llama_sampler_i {
    name: Some(name),
    accept: Some(accept),
    apply: Some(apply),
    reset: Some(reset),
    clone: Some(clone),
    free: Some(free),
    backend_init: None,
    backend_accept: None,
    backend_apply: None,
    backend_set_input: None,
    backend_reset: None,
    copy_state: None,
};

fn abort(message: &str) -> ! {
    let message = CString::new(message).unwrap_or_default();
    let file = CString::new(file!()).unwrap_or_default();
    // SAFETY: ggml_abort formats "%s" with a valid C string and never returns.
    unsafe {
        ggml_abort(
            file.as_ptr(),
            line!() as i32,
            c"%s".as_ptr(),
            message.as_ptr(),
        )
    };
    std::process::abort()
}

fn error_text(text: *const c_char) -> String {
    if text.is_null() {
        return String::new();
    }
    // SAFETY: llguidance returns a NUL-terminated string owned by the matcher.
    unsafe { CStr::from_ptr(text) }
        .to_string_lossy()
        .into_owned()
}

fn new_matcher(tokenizer: *mut LlgTokenizer, kind: &CStr, data: &CStr) -> *mut LlgMatcher {
    // SAFETY: LlgConstraintInit is plain data that llg_constraint_init_set_defaults fills in.
    let mut init: LlgConstraintInit = unsafe { std::mem::zeroed() };
    llg_constraint_init_set_defaults(&mut init, tokenizer);
    if let Some(level) = std::env::var_os("LLGUIDANCE_LOG_LEVEL").filter(|v| !v.is_empty()) {
        let level = CString::new(level.into_encoded_bytes()).unwrap_or_default();
        // SAFETY: atoi reads a NUL-terminated string.
        init.log_stderr_level = unsafe { libc::atoi(level.as_ptr()) } as u32;
    }
    // SAFETY: init holds a live tokenizer and both strings are NUL-terminated.
    let matcher = unsafe { llg_new_matcher(&init, kind.as_ptr(), data.as_ptr()) };
    // SAFETY: llg_new_matcher never returns null.
    let error = llg_matcher_get_error(unsafe { &mut *matcher });
    if !error.is_null() {
        log::emit(
            log::ERROR,
            FUNC,
            &format!("llg error: {}\n", error_text(error)),
        );
        // SAFETY: matcher came from llg_new_matcher and is freed once.
        unsafe { llg_free_matcher(matcher) };
        return ptr::null_mut();
    }
    matcher
}

extern "C" fn tokenize(
    user_data: *const c_void,
    bytes: *const u8,
    bytes_len: usize,
    output_tokens: *mut u32,
    output_tokens_len: usize,
) -> usize {
    // SAFETY: user_data is the vocab passed in LlgTokenizerInit; the buffers come from llguidance.
    let r = unsafe {
        llama_tokenize(
            user_data.cast(),
            bytes.cast(),
            bytes_len as i32,
            output_tokens.cast(),
            output_tokens_len as i32,
            false,
            true,
        )
    };
    r.unsigned_abs() as usize
}

struct TokenizerCache {
    vocab: *const llama_vocab,
    tokenizer: *mut LlgTokenizer,
}

// SAFETY: the cache is only reached through the mutex, and llguidance tokenizers are Send.
unsafe impl Send for TokenizerCache {}

static CACHE: Mutex<TokenizerCache> = Mutex::new(TokenizerCache {
    vocab: ptr::null(),
    tokenizer: ptr::null_mut(),
});

fn new_tokenizer(vocab: *const llama_vocab) -> *mut LlgTokenizer {
    let mut cache = CACHE.lock().unwrap_or_else(|e| e.into_inner());
    if !cache.vocab.is_null() && cache.vocab == vocab {
        // SAFETY: the cached tokenizer stays alive until it is replaced.
        return llg_clone_tokenizer(unsafe { &*cache.tokenizer });
    }

    // SAFETY: vocab is a live vocabulary for the whole call.
    let (tok_eos, vocab_size) = unsafe {
        let eot = llama_vocab_eot(vocab);
        let eos = if eot == LLAMA_TOKEN_NULL {
            llama_vocab_eos(vocab)
        } else {
            eot
        };
        (eos, llama_vocab_n_tokens(vocab) as usize)
    };

    const MAX_TOKEN: usize = 1024;
    let mut token_lens = vec![0u32; vocab_size];
    let mut token_bytes = vec![0u8; vocab_size * 16 + 1024 * 1024];
    let mut offset = 0;
    for (i, len) in token_lens.iter_mut().enumerate() {
        if token_bytes.len() - offset < MAX_TOKEN {
            abort("token_bytes buffer too small\n");
        }
        let token = i as llama_token;
        let dp = token_bytes[offset..].as_mut_ptr();
        // SAFETY: dp has at least MAX_TOKEN writable bytes.
        let mut size = unsafe {
            llama_detokenize(vocab, &token, 1, dp.cast(), MAX_TOKEN as i32, false, false)
        };
        if size < 0 {
            abort("llama_detokenize failed\n");
        }
        if size == 0 {
            // SAFETY: as above, one byte further in.
            size = unsafe {
                llama_detokenize(
                    vocab,
                    &token,
                    1,
                    dp.add(1).cast(),
                    MAX_TOKEN as i32 - 1,
                    false,
                    true,
                )
            };
            if size < 0 {
                abort("llama_detokenize failed\n");
            }
            if size != 0 {
                token_bytes[offset] = 0xff;
                size += 1;
            }
        }
        *len = size as u32;
        offset += size as usize;
    }

    let tinit = LlgTokenizerInit {
        vocab_size: vocab_size as u32,
        tok_eos: tok_eos as u32,
        token_lens: token_lens.as_ptr(),
        token_bytes: token_bytes.as_ptr(),
        tokenizer_json: ptr::null(),
        tokenize_assumes_string: true,
        tokenize_fn: Some(tokenize),
        use_approximate_greedy_tokenize_fn: false,
        tokenize_user_data: vocab.cast(),
        slices: ptr::null(),
    };

    let mut error = [0 as c_char; 1024];
    // SAFETY: tinit points at buffers that outlive the call; error has the length passed.
    let tokenizer = unsafe { llg_new_tokenizer(&tinit, error.as_mut_ptr(), error.len()) };
    if tokenizer.is_null() {
        log::emit(
            log::ERROR,
            FUNC,
            &format!("llg tokenizer error: {}\n", error_text(error.as_ptr())),
        );
        return tokenizer;
    }

    if !cache.tokenizer.is_null() {
        // SAFETY: the cached tokenizer is owned by the cache and freed once.
        unsafe { llg_free_tokenizer(cache.tokenizer) };
    }
    cache.vocab = vocab;
    cache.tokenizer = tokenizer;
    // SAFETY: just stored above.
    llg_clone_tokenizer(unsafe { &*tokenizer })
}

fn new_sampler(ctx: Ctx) -> *mut llama_sampler {
    // SAFETY: IFACE is never written through the pointer; libllama only reads the vtable.
    unsafe {
        llama_sampler_init(
            ptr::addr_of!(IFACE).cast_mut(),
            Box::into_raw(Box::new(ctx)).cast(),
        )
    }
}

unsafe fn ctx<'a>(smpl: *const llama_sampler) -> &'a mut Ctx {
    // SAFETY: every sampler with IFACE was created by new_sampler, so ctx is a live Box<Ctx>.
    unsafe { &mut *(*smpl).ctx.cast::<Ctx>() }
}

unsafe extern "C" fn name(_: *const llama_sampler) -> *const c_char {
    c"llguidance".as_ptr()
}

unsafe extern "C" fn accept(smpl: *mut llama_sampler, token: llama_token) {
    // SAFETY: called by libllama on a sampler built with IFACE.
    let ctx = unsafe { ctx(smpl) };
    if !ctx.grammar.is_null() {
        // SAFETY: grammar is a live matcher owned by ctx.
        llg_matcher_consume_token(unsafe { &mut *ctx.grammar }, token as u32);
    }
}

unsafe extern "C" fn apply(smpl: *mut llama_sampler, cur_p: *mut llama_token_data_array) {
    // SAFETY: called by libllama on a sampler built with IFACE and a valid candidate array.
    let (ctx, cur) = unsafe { (ctx(smpl), &*cur_p) };
    if ctx.grammar.is_null() {
        return;
    }
    // SAFETY: grammar is a live matcher owned by ctx.
    let grammar = unsafe { &mut *ctx.grammar };
    let mut mask = llg_matcher_get_mask(grammar);
    if mask.is_null() {
        if llg_matcher_compute_mask(grammar) == 0 {
            mask = llg_matcher_get_mask(grammar);
        } else {
            let error = error_text(llg_matcher_get_error(grammar));
            log::emit(log::ERROR, FUNC, &format!("llg error: {error}\n"));
            // SAFETY: the matcher is owned by ctx and dropped here once.
            unsafe { llg_free_matcher(ctx.grammar) };
            ctx.grammar = ptr::null_mut();
            return;
        }
    }
    if cur.data.is_null() {
        return;
    }
    // SAFETY: the mask covers the vocabulary (checked at init); data holds size candidates.
    let (mask, cur) = unsafe {
        (
            slice::from_raw_parts(mask, llg_matcher_get_mask_byte_size(grammar) / 4),
            slice::from_raw_parts_mut(cur.data, cur.size),
        )
    };
    for td in cur {
        let token = td.id as u32;
        if mask[(token / 32) as usize] & (1u32 << (token % 32)) == 0 {
            td.logit = f32::NEG_INFINITY;
        }
    }
}

unsafe extern "C" fn reset(smpl: *mut llama_sampler) {
    // SAFETY: called by libllama on a sampler built with IFACE.
    let ctx = unsafe { ctx(smpl) };
    if !ctx.grammar.is_null() {
        // SAFETY: grammar is a live matcher owned by ctx.
        llg_matcher_reset(unsafe { &mut *ctx.grammar });
    }
}

unsafe extern "C" fn clone(smpl: *const llama_sampler) -> *mut llama_sampler {
    // SAFETY: called by libllama on a sampler built with IFACE.
    let ctx = unsafe { ctx(smpl) };
    let mut copy = Ctx {
        vocab: ctx.vocab,
        grammar_kind: CString::default(),
        grammar_data: CString::default(),
        tokenizer: ptr::null_mut(),
        grammar: ptr::null_mut(),
    };
    if !ctx.grammar.is_null() {
        copy.grammar_kind = ctx.grammar_kind.clone();
        copy.grammar_data = ctx.grammar_data.clone();
        // SAFETY: both are live and owned by ctx.
        unsafe {
            copy.grammar = llg_clone_matcher(&*ctx.grammar);
            copy.tokenizer = llg_clone_tokenizer(&*ctx.tokenizer);
        }
    }
    new_sampler(copy)
}

unsafe extern "C" fn free(smpl: *mut llama_sampler) {
    // SAFETY: ctx came from Box::into_raw in new_sampler and libllama frees each sampler once.
    let ctx = unsafe { Box::from_raw((*smpl).ctx.cast::<Ctx>()) };
    if !ctx.grammar.is_null() {
        // SAFETY: both are owned by ctx and freed once.
        unsafe {
            llg_free_matcher(ctx.grammar);
            llg_free_tokenizer(ctx.tokenizer);
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn llama_rs_sampler_init_llg(
    vocab: *const llama_vocab,
    grammar_kind: *const c_char,
    grammar_data: *const c_char,
) -> *mut llama_sampler {
    let mut ctx = Ctx {
        vocab,
        grammar_kind: CString::default(),
        grammar_data: CString::default(),
        tokenizer: ptr::null_mut(),
        grammar: ptr::null_mut(),
    };
    // SAFETY: the caller passes NUL-terminated strings or null.
    let kind = (!grammar_kind.is_null()).then(|| unsafe { CStr::from_ptr(grammar_kind) });
    if let Some(kind) = kind.filter(|k| !k.is_empty()) {
        let data = if grammar_data.is_null() {
            CString::default()
        } else {
            // SAFETY: as above.
            unsafe { CStr::from_ptr(grammar_data) }.to_owned()
        };
        ctx.tokenizer = new_tokenizer(vocab);
        ctx.grammar = new_matcher(ctx.tokenizer, kind, &data);
        ctx.grammar_kind = kind.to_owned();
        ctx.grammar_data = data;
        if !ctx.grammar.is_null() {
            // SAFETY: vocab is live; grammar was just created.
            let n_tokens = unsafe { llama_vocab_n_tokens(vocab) } as usize;
            if n_tokens.div_ceil(32) * 4
                != llg_matcher_get_mask_byte_size(unsafe { &mut *ctx.grammar })
            {
                abort(
                    "GGML_ASSERT(((size_t) llama_vocab_n_tokens(vocab) + 31) / 32 * 4 == llg_matcher_get_mask_byte_size(ctx->grammar)) failed",
                );
            }
        }
    }
    new_sampler(ctx)
}
