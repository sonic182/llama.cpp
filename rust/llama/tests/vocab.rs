use std::path::PathBuf;

use llama::{Backend, Model};

#[test]
fn vocab_only_model_tokenizes_and_links_shim() {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/ggml-vocab-llama-bpe.gguf");

    let backend = Backend::init();
    let model = Model::load(&backend, &path, true).unwrap();
    let vocab = model.vocab();

    assert_eq!(vocab.n_tokens(), 128256);

    let tokens = vocab.tokenize(b"Hello world", false, false).unwrap();
    assert_eq!(tokens, [9906, 1917]);
    assert_eq!(
        vocab.detokenize(&tokens, false, false).unwrap(),
        b"Hello world"
    );

    assert!(matches!(
        vocab.token_to_piece(-1, true),
        Err(llama::Error::InvalidToken)
    ));
    assert!(matches!(
        vocab.detokenize(&[9906, 128256], false, false),
        Err(llama::Error::InvalidToken)
    ));

    assert_eq!(
        unsafe { llama::sys::llama_ext_c_model_n_expert(model.as_ptr()) },
        0
    );
}
