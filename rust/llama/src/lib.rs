use std::ffi::CString;
use std::fmt;
use std::marker::PhantomData;
use std::os::unix::ffi::OsStrExt;
use std::path::Path;
use std::ptr::{self, NonNull};

pub mod gguf;

pub use llama_sys as sys;

#[derive(Debug)]
pub enum Error {
    InvalidPath,
    ModelLoad,
    GgufLoad,
    TextTooLong,
    InvalidToken,
    Tokenize,
    Detokenize,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let msg = match self {
            Error::InvalidPath => "path contains a NUL byte",
            Error::ModelLoad => "failed to load model",
            Error::GgufLoad => "failed to load GGUF file",
            Error::TextTooLong => "text is too long",
            Error::InvalidToken => "token id is out of range",
            Error::Tokenize => "tokenization failed",
            Error::Detokenize => "detokenization failed",
        };
        f.write_str(msg)
    }
}

impl std::error::Error for Error {}

pub struct Backend(());

impl Backend {
    pub fn init() -> Self {
        unsafe { sys::llama_backend_init() };
        Backend(())
    }
}

impl Drop for Backend {
    fn drop(&mut self) {
        unsafe { sys::llama_backend_free() };
    }
}

pub struct Model<'b> {
    ptr: NonNull<sys::llama_model>,
    _backend: PhantomData<&'b Backend>,
}

impl<'b> Model<'b> {
    pub fn load(_backend: &'b Backend, path: &Path, vocab_only: bool) -> Result<Self, Error> {
        let path = CString::new(path.as_os_str().as_bytes()).map_err(|_| Error::InvalidPath)?;
        let mut params = unsafe { sys::llama_model_default_params() };
        params.vocab_only = vocab_only;
        let ptr = unsafe { sys::llama_model_load_from_file(path.as_ptr(), params) };
        let ptr = NonNull::new(ptr).ok_or(Error::ModelLoad)?;
        Ok(Model {
            ptr,
            _backend: PhantomData,
        })
    }

    pub fn as_ptr(&self) -> *const sys::llama_model {
        self.ptr.as_ptr()
    }

    pub fn vocab(&self) -> Vocab<'_> {
        let ptr = unsafe { sys::llama_model_get_vocab(self.ptr.as_ptr()) };
        Vocab {
            ptr: NonNull::new(ptr.cast_mut()).expect("model has no vocab"),
            _model: PhantomData,
        }
    }
}

impl Drop for Model<'_> {
    fn drop(&mut self) {
        unsafe { sys::llama_model_free(self.ptr.as_ptr()) };
    }
}

pub struct Vocab<'m> {
    ptr: NonNull<sys::llama_vocab>,
    _model: PhantomData<&'m ()>,
}

impl Vocab<'_> {
    pub fn n_tokens(&self) -> usize {
        unsafe { sys::llama_vocab_n_tokens(self.ptr.as_ptr()) as usize }
    }

    pub fn add_bos(&self) -> bool {
        unsafe { sys::llama_vocab_get_add_bos(self.ptr.as_ptr()) }
    }

    fn check_tokens(&self, tokens: &[sys::llama_token]) -> Result<(), Error> {
        let n = self.n_tokens();
        if tokens
            .iter()
            .all(|&t| usize::try_from(t).is_ok_and(|t| t < n))
        {
            Ok(())
        } else {
            Err(Error::InvalidToken)
        }
    }

    pub fn token_to_piece(&self, token: sys::llama_token, special: bool) -> Result<Vec<u8>, Error> {
        self.check_tokens(&[token])?;
        let mut piece: Vec<u8> = Vec::with_capacity(32);
        let mut n = unsafe {
            sys::llama_token_to_piece(
                self.ptr.as_ptr(),
                token,
                piece.as_mut_ptr().cast(),
                piece.capacity() as i32,
                0,
                special,
            )
        };
        if n < 0 {
            piece.reserve_exact(n.unsigned_abs() as usize);
            n = unsafe {
                sys::llama_token_to_piece(
                    self.ptr.as_ptr(),
                    token,
                    piece.as_mut_ptr().cast(),
                    piece.capacity() as i32,
                    0,
                    special,
                )
            };
        }
        if n < 0 {
            return Err(Error::Detokenize);
        }
        unsafe { piece.set_len(n as usize) };
        Ok(piece)
    }

    pub fn tokenize(
        &self,
        text: &[u8],
        add_special: bool,
        parse_special: bool,
    ) -> Result<Vec<sys::llama_token>, Error> {
        let len = i32::try_from(text.len()).map_err(|_| Error::TextTooLong)?;
        let needed = unsafe {
            sys::llama_tokenize(
                self.ptr.as_ptr(),
                text.as_ptr().cast(),
                len,
                ptr::null_mut(),
                0,
                add_special,
                parse_special,
            )
        };
        let cap = needed.checked_abs().ok_or(Error::Tokenize)?;
        let mut tokens: Vec<sys::llama_token> = Vec::with_capacity(cap as usize);
        let n = unsafe {
            sys::llama_tokenize(
                self.ptr.as_ptr(),
                text.as_ptr().cast(),
                len,
                tokens.as_mut_ptr(),
                cap,
                add_special,
                parse_special,
            )
        };
        if n < 0 {
            return Err(Error::Tokenize);
        }
        unsafe { tokens.set_len(n as usize) };
        Ok(tokens)
    }

    pub fn detokenize(
        &self,
        tokens: &[sys::llama_token],
        remove_special: bool,
        unparse_special: bool,
    ) -> Result<Vec<u8>, Error> {
        self.check_tokens(tokens)?;
        let n_tokens = i32::try_from(tokens.len()).map_err(|_| Error::TextTooLong)?;
        let needed = unsafe {
            sys::llama_detokenize(
                self.ptr.as_ptr(),
                tokens.as_ptr(),
                n_tokens,
                ptr::null_mut(),
                0,
                remove_special,
                unparse_special,
            )
        };
        let cap = needed.checked_abs().ok_or(Error::Detokenize)?;
        let mut text: Vec<u8> = Vec::with_capacity(cap as usize);
        let n = unsafe {
            sys::llama_detokenize(
                self.ptr.as_ptr(),
                tokens.as_ptr(),
                n_tokens,
                text.as_mut_ptr().cast(),
                cap,
                remove_special,
                unparse_special,
            )
        };
        if n < 0 {
            return Err(Error::Detokenize);
        }
        unsafe { text.set_len(n as usize) };
        Ok(text)
    }
}

pub fn version() -> &'static str {
    unsafe { std::ffi::CStr::from_ptr(sys::llama_version()) }
        .to_str()
        .unwrap_or("unknown")
}
