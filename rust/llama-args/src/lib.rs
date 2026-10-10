//! Command line arguments and the Params they fill, ported from common/arg.cpp.

pub mod ffi;
pub mod models;
pub mod params;

pub use params::Params;

pub fn params_from_cbor(data: &[u8]) -> Result<Params, String> {
    ciborium::from_reader(data).map_err(|e| e.to_string())
}

pub fn params_to_cbor(params: &Params) -> Vec<u8> {
    let mut out = Vec::new();
    ciborium::into_writer(params, &mut out).expect("writing to a Vec cannot fail");
    out
}
