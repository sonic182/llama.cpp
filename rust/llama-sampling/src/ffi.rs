use std::ffi::c_char;
use std::ptr;
use std::slice;

use llama::sys::{
    llama_sampler, llama_sampler_i, llama_sampler_init, llama_token, llama_token_data_array,
    llama_token_to_piece, llama_vocab,
};

use crate::budget::{Budget, State, utf8_is_complete};
use crate::log;

struct Ctx {
    vocab: *const llama_vocab,
    budget: Budget,
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

fn token_piece(vocab: *const llama_vocab, token: llama_token) -> Vec<u8> {
    let mut buf = vec![0u8; 16];
    // SAFETY: vocab is non-null and buf has the length passed.
    let n = unsafe { llama_token_to_piece(vocab, token, buf.as_mut_ptr().cast(), 16, 0, false) };
    if n < 0 {
        buf.resize(n.unsigned_abs() as usize, 0);
        // SAFETY: as above, with the size libllama asked for.
        let check =
            unsafe { llama_token_to_piece(vocab, token, buf.as_mut_ptr().cast(), -n, 0, false) };
        assert_eq!(check, -n);
    } else {
        buf.truncate(n as usize);
    }
    buf
}

unsafe extern "C" fn name(_: *const llama_sampler) -> *const c_char {
    c"reasoning-budget".as_ptr()
}

unsafe extern "C" fn accept(smpl: *mut llama_sampler, token: llama_token) {
    // SAFETY: called by libllama on a sampler built with IFACE.
    let ctx = unsafe { ctx(smpl) };
    let vocab = ctx.vocab;
    ctx.budget.accept(token, || {
        vocab.is_null() || utf8_is_complete(&token_piece(vocab, token))
    });
}

unsafe extern "C" fn apply(smpl: *mut llama_sampler, cur_p: *mut llama_token_data_array) {
    // SAFETY: called by libllama on a sampler built with IFACE and a valid candidate array.
    let (ctx, cur) = unsafe { (ctx(smpl), &*cur_p) };
    if cur.data.is_null() {
        return;
    }
    // SAFETY: data points to size candidates owned by the caller.
    ctx.budget
        .apply(unsafe { slice::from_raw_parts_mut(cur.data, cur.size) });
}

unsafe extern "C" fn reset(smpl: *mut llama_sampler) {
    // SAFETY: called by libllama on a sampler built with IFACE.
    unsafe { ctx(smpl) }.budget.reset();
}

unsafe extern "C" fn clone(smpl: *const llama_sampler) -> *mut llama_sampler {
    // SAFETY: called by libllama on a sampler built with IFACE.
    let ctx = unsafe { ctx(smpl) };
    new_sampler(Ctx {
        vocab: ctx.vocab,
        budget: ctx.budget.clone(),
    })
}

unsafe extern "C" fn free(smpl: *mut llama_sampler) {
    // SAFETY: ctx came from Box::into_raw in new_sampler and libllama frees each sampler once.
    drop(unsafe { Box::from_raw((*smpl).ctx.cast::<Ctx>()) });
}

unsafe fn seqs(tokens: *const llama_token, lens: *const usize, n: usize) -> Vec<Vec<llama_token>> {
    if n == 0 {
        return Vec::new();
    }
    // SAFETY: the caller passes n lengths and their sum of tokens.
    let lens = unsafe { slice::from_raw_parts(lens, n) };
    let mut offset = 0;
    lens.iter()
        .map(|&len| {
            let seq = if len == 0 {
                Vec::new()
            } else {
                // SAFETY: as above.
                unsafe { slice::from_raw_parts(tokens.add(offset), len) }.to_vec()
            };
            offset += len;
            seq
        })
        .collect()
}

fn as_budget<'a>(smpl: *const llama_sampler) -> Option<&'a mut Budget> {
    if smpl.is_null() {
        return None;
    }
    // SAFETY: a non-null sampler passed to these functions was created by
    // llama_rs_reasoning_budget_init.
    Some(&mut unsafe { ctx(smpl) }.budget)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn llama_rs_reasoning_budget_init(
    vocab: *const llama_vocab,
    start_tokens: *const llama_token,
    start_lens: *const usize,
    n_start: usize,
    end_tokens: *const llama_token,
    end_lens: *const usize,
    n_end: usize,
    forced_tokens: *const llama_token,
    n_forced: usize,
    budget: i32,
    initial_state: i32,
) -> *mut llama_sampler {
    // SAFETY: the caller passes valid arrays with the given sizes.
    let (start, end, forced) = unsafe {
        (
            seqs(start_tokens, start_lens, n_start),
            seqs(end_tokens, end_lens, n_end),
            if n_forced == 0 {
                Vec::new()
            } else {
                slice::from_raw_parts(forced_tokens, n_forced).to_vec()
            },
        )
    };
    let state = State::from_raw(initial_state).unwrap_or(State::Idle);
    new_sampler(Ctx {
        vocab,
        budget: Budget::new(&start, &end, forced, budget, state),
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn llama_rs_reasoning_budget_get_state(smpl: *const llama_sampler) -> i32 {
    as_budget(smpl).map_or(State::Idle, |b| b.state()) as i32
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn llama_rs_reasoning_budget_get_end_match(
    smpl: *const llama_sampler,
    n_tokens: *mut usize,
) -> *const llama_token {
    let matched = as_budget(smpl).and_then(|b| b.end_match());
    if !n_tokens.is_null() {
        // SAFETY: the caller passes a valid out pointer or null.
        unsafe { *n_tokens = matched.map_or(0, <[_]>::len) };
    }
    matched.map_or(ptr::null(), <[_]>::as_ptr)
}

#[unsafe(no_mangle)]
pub extern "C" fn llama_rs_reasoning_budget_force(smpl: *mut llama_sampler) -> bool {
    as_budget(smpl).is_some_and(Budget::force)
}

#[unsafe(no_mangle)]
pub extern "C" fn llama_rs_sampling_set_log_sink(sink: log::Sink) {
    log::set_sink(sink);
}
