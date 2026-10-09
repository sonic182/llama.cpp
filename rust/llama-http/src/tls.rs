use std::io;

use hyper_util::rt::TokioIo;
use tokio::io::{AsyncRead, AsyncWrite};

#[cfg(feature = "tls")]
pub struct Tls(tokio_rustls::TlsAcceptor);

#[cfg(not(feature = "tls"))]
pub enum Tls {}

#[cfg(feature = "tls")]
impl Tls {
    pub fn load(cert_path: &str, key_path: &str) -> io::Result<Self> {
        use std::sync::Arc;

        use rustls::{ServerConfig, crypto::ring};
        use rustls_pki_types::{CertificateDer, PrivateKeyDer, pem::PemObject};

        let invalid = |path: &str, err: &dyn std::fmt::Display| {
            io::Error::new(io::ErrorKind::InvalidData, format!("{path}: {err}"))
        };

        let certs = CertificateDer::pem_file_iter(cert_path)
            .map_err(|err| invalid(cert_path, &err))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|err| invalid(cert_path, &err))?;
        if certs.is_empty() {
            return Err(invalid(cert_path, &"no certificate found"));
        }
        let key = PrivateKeyDer::from_pem_file(key_path).map_err(|err| invalid(key_path, &err))?;

        let mut config = ServerConfig::builder_with_provider(Arc::new(ring::default_provider()))
            .with_safe_default_protocol_versions()
            .map_err(io::Error::other)?
            .with_no_client_auth()
            .with_single_cert(certs, key)
            .map_err(|err| invalid(key_path, &err))?;
        config.alpn_protocols = vec![b"http/1.1".to_vec()];
        Ok(Self(tokio_rustls::TlsAcceptor::from(Arc::new(config))))
    }

    pub async fn accept<S>(
        &self,
        stream: S,
    ) -> io::Result<TokioIo<tokio_rustls::server::TlsStream<S>>>
    where
        S: AsyncRead + AsyncWrite + Unpin,
    {
        self.0.accept(stream).await.map(TokioIo::new)
    }
}

#[cfg(not(feature = "tls"))]
impl Tls {
    pub fn load(_cert_path: &str, _key_path: &str) -> io::Result<Self> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "the server is built without SSL support",
        ))
    }

    pub async fn accept<S>(&self, _stream: S) -> io::Result<TokioIo<S>>
    where
        S: AsyncRead + AsyncWrite + Unpin,
    {
        match *self {}
    }
}
