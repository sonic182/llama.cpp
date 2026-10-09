use std::ffi::{CStr, CString};
use std::os::unix::ffi::OsStrExt;
use std::path::Path;
use std::ptr::{self, NonNull};

use crate::{Error, sys};

const SPLIT_PATH_MAX: usize = 4096;

pub struct GgufFile {
    ctx: NonNull<sys::gguf_context>,
    meta: NonNull<sys::ggml_context>,
}

impl GgufFile {
    pub fn open(path: &Path) -> Result<Self, Error> {
        Self::open_impl(path, true)
    }

    pub fn open_with_data(path: &Path) -> Result<Self, Error> {
        Self::open_impl(path, false)
    }

    fn open_impl(path: &Path, no_alloc: bool) -> Result<Self, Error> {
        let path = CString::new(path.as_os_str().as_bytes()).map_err(|_| Error::InvalidPath)?;
        let mut meta: *mut sys::ggml_context = ptr::null_mut();
        let params = sys::gguf_init_params {
            no_alloc,
            ctx: &mut meta,
        };
        let ctx = unsafe { sys::gguf_init_from_file(path.as_ptr(), params) };
        match (NonNull::new(ctx), NonNull::new(meta)) {
            (Some(ctx), Some(meta)) => Ok(GgufFile { ctx, meta }),
            (Some(ctx), None) => {
                unsafe { sys::gguf_free(ctx.as_ptr()) };
                Err(Error::GgufLoad)
            }
            (None, Some(meta)) => {
                unsafe { sys::ggml_free(meta.as_ptr()) };
                Err(Error::GgufLoad)
            }
            (None, None) => Err(Error::GgufLoad),
        }
    }

    pub fn n_tensors(&self) -> usize {
        unsafe { sys::gguf_get_n_tensors(self.ctx.as_ptr()) as usize }
    }

    pub fn tensor_name(&self, index: usize) -> &CStr {
        unsafe { CStr::from_ptr(sys::gguf_get_tensor_name(self.ctx.as_ptr(), index as i64)) }
    }

    pub fn tensor_nbytes(&self, index: usize) -> usize {
        unsafe { sys::ggml_nbytes(self.tensor(index)) }
    }

    pub fn tensor_offset(&self, index: usize) -> usize {
        unsafe { sys::gguf_get_tensor_offset(self.ctx.as_ptr(), index as i64) }
    }

    pub fn data_offset(&self) -> usize {
        unsafe { sys::gguf_get_data_offset(self.ctx.as_ptr()) }
    }

    pub fn find_tensor(&self, name: &CStr) -> Option<usize> {
        let index = unsafe { sys::gguf_find_tensor(self.ctx.as_ptr(), name.as_ptr()) };
        usize::try_from(index).ok()
    }

    pub fn find_key(&self, key: &CStr) -> Option<usize> {
        let id = unsafe { sys::gguf_find_key(self.ctx.as_ptr(), key.as_ptr()) };
        usize::try_from(id).ok()
    }

    pub fn val_u16(&self, key_id: usize) -> u16 {
        unsafe { sys::gguf_get_val_u16(self.ctx.as_ptr(), key_id as i64) }
    }

    pub fn set_u16(&mut self, key: &CStr, value: u16) {
        unsafe { sys::gguf_set_val_u16(self.ctx.as_ptr(), key.as_ptr(), value) };
    }

    pub fn kv_type(&self, key_id: usize) -> sys::gguf_type::Type {
        unsafe { sys::gguf_get_kv_type(self.ctx.as_ptr(), key_id as i64) }
    }

    pub fn arr_type(&self, key_id: usize) -> sys::gguf_type::Type {
        unsafe { sys::gguf_get_arr_type(self.ctx.as_ptr(), key_id as i64) }
    }

    pub fn arr_n(&self, key_id: usize) -> usize {
        unsafe { sys::gguf_get_arr_n(self.ctx.as_ptr(), key_id as i64) }
    }

    pub fn arr_str(&self, key_id: usize, index: usize) -> &CStr {
        unsafe {
            CStr::from_ptr(sys::gguf_get_arr_str(
                self.ctx.as_ptr(),
                key_id as i64,
                index,
            ))
        }
    }

    pub fn val_u32(&self, key_id: usize) -> u32 {
        unsafe { sys::gguf_get_val_u32(self.ctx.as_ptr(), key_id as i64) }
    }

    pub fn meta_tensors(&self) -> impl Iterator<Item = Tensor<'_>> {
        let first = unsafe { sys::ggml_get_first_tensor(self.meta.as_ptr()) };
        std::iter::successors(NonNull::new(first), move |t| {
            NonNull::new(unsafe { sys::ggml_get_next_tensor(self.meta.as_ptr(), t.as_ptr()) })
        })
        .map(|ptr| Tensor {
            ptr,
            _file: std::marker::PhantomData,
        })
    }

    fn tensor(&self, index: usize) -> *mut sys::ggml_tensor {
        let tensor =
            unsafe { sys::ggml_get_tensor(self.meta.as_ptr(), self.tensor_name(index).as_ptr()) };
        assert!(!tensor.is_null(), "gguf meta context lists every tensor");
        tensor
    }
}

impl Drop for GgufFile {
    fn drop(&mut self) {
        unsafe {
            sys::gguf_free(self.ctx.as_ptr());
            sys::ggml_free(self.meta.as_ptr());
        }
    }
}

pub struct Tensor<'a> {
    ptr: NonNull<sys::ggml_tensor>,
    _file: std::marker::PhantomData<&'a GgufFile>,
}

impl Tensor<'_> {
    pub fn name(&self) -> &CStr {
        unsafe { CStr::from_ptr((*self.ptr.as_ptr()).name.as_ptr()) }
    }

    pub fn ty(&self) -> sys::ggml_type::Type {
        unsafe { (*self.ptr.as_ptr()).type_ }
    }

    pub fn f32_data(&self) -> Option<&[f32]> {
        let t = self.ptr.as_ptr();
        unsafe {
            if (*t).type_ != sys::ggml_type::GGML_TYPE_F32 || (*t).data.is_null() {
                return None;
            }
            let n = usize::try_from(sys::ggml_nelements(t)).ok()?;
            Some(std::slice::from_raw_parts((*t).data.cast::<f32>(), n))
        }
    }
}

pub struct GgufBuilder {
    ctx: NonNull<sys::gguf_context>,
}

impl GgufBuilder {
    pub fn new() -> Self {
        let ctx = unsafe { sys::gguf_init_empty() };
        GgufBuilder {
            ctx: NonNull::new(ctx).expect("gguf_init_empty allocates"),
        }
    }

    pub fn copy_kv(&mut self, src: &GgufFile) {
        unsafe { sys::gguf_set_kv(self.ctx.as_ptr(), src.ctx.as_ptr()) };
    }

    pub fn set_u16(&mut self, key: &CStr, value: u16) {
        unsafe { sys::gguf_set_val_u16(self.ctx.as_ptr(), key.as_ptr(), value) };
    }

    pub fn set_i32(&mut self, key: &CStr, value: i32) {
        unsafe { sys::gguf_set_val_i32(self.ctx.as_ptr(), key.as_ptr(), value) };
    }

    pub fn add_tensor(&mut self, src: &GgufFile, index: usize) {
        unsafe { sys::gguf_add_tensor(self.ctx.as_ptr(), src.tensor(index)) };
    }

    pub fn n_tensors(&self) -> usize {
        unsafe { sys::gguf_get_n_tensors(self.ctx.as_ptr()) as usize }
    }

    pub fn tensor_name(&self, index: usize) -> &CStr {
        unsafe { CStr::from_ptr(sys::gguf_get_tensor_name(self.ctx.as_ptr(), index as i64)) }
    }

    pub fn meta_size(&self) -> usize {
        unsafe { sys::gguf_get_meta_size(self.ctx.as_ptr()) }
    }

    pub fn meta_data(&self) -> Vec<u8> {
        let mut data = vec![0u8; self.meta_size()];
        unsafe { sys::gguf_get_meta_data(self.ctx.as_ptr(), data.as_mut_ptr().cast()) };
        data
    }
}

impl Default for GgufBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for GgufBuilder {
    fn drop(&mut self) {
        unsafe { sys::gguf_free(self.ctx.as_ptr()) };
    }
}

pub fn split_path(prefix: &[u8], split_no: i32, split_count: i32) -> Option<Vec<u8>> {
    let prefix = CString::new(prefix).ok()?;
    let mut buf = vec![0u8; SPLIT_PATH_MAX];
    let written = unsafe {
        sys::llama_split_path(
            buf.as_mut_ptr().cast(),
            buf.len(),
            prefix.as_ptr(),
            split_no,
            split_count,
        )
    };
    let written = usize::try_from(written).ok().filter(|&n| n > 0)?;
    buf.truncate(written);
    Some(buf)
}

pub fn split_prefix(path: &[u8], split_no: i32, split_count: i32) -> Option<Vec<u8>> {
    let path = CString::new(path).ok()?;
    let mut buf = vec![0u8; SPLIT_PATH_MAX];
    let written = unsafe {
        sys::llama_split_prefix(
            buf.as_mut_ptr().cast(),
            buf.len(),
            path.as_ptr(),
            split_no,
            split_count,
        )
    };
    let written = usize::try_from(written).ok().filter(|&n| n > 0)?;
    buf.truncate(written);
    Some(buf)
}
