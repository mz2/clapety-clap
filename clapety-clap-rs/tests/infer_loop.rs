use clapety_clap::{expand_audio_files, InferenceConfig, TagInferencer};
use std::path::PathBuf;

#[test]
fn infer_on_fixture_loop() {
    let fixture = PathBuf::from("../tests/fixtures/loop.wav");
    assert!(fixture.exists(), "fixture missing: {:?}", fixture);
    let files = expand_audio_files(vec![fixture.clone()]).expect("expand");
    assert_eq!(files.len(), 1);
    let infer = TagInferencer::default();
    let cfg = InferenceConfig { top_k: 5 };
    let out = infer.infer_paths(files, &cfg).expect("infer");
    assert_eq!(out.len(), 1);
    let fc = &out[0];
    assert_eq!(fc.tags.len(), 5);
    assert!(!fc.caption.is_empty());
    assert!(fc.caption.split(',').count() <= 5);
}

#[cfg(feature = "onnx")]
#[test]
fn onnx_dim_matches_exported() {
    // Skip unless user provides model path via env to avoid CI failures.
    let audio = std::env::var("CLAP_ONNX_AUDIO").ok();
    let text = std::env::var("CLAP_ONNX_TEXT").ok();
    if audio.is_none() || text.is_none() {
        eprintln!("Skipping onnx_dim_matches_exported (set CLAP_ONNX_AUDIO & CLAP_ONNX_TEXT)");
        return;
    }
    let backend = clapety_clap::OnnxBackend::from_paths(
        audio.as_ref().unwrap().as_ref(),
        text.as_ref().unwrap().as_ref(),
    )
    .expect("backend");
    let a = backend.embed_audio(&[]);
    assert!(a.len() > 0);
}
