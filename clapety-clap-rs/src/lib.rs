//! clapety-clap Rust core library
//!
//! Placeholder implementation that mimics the Python interface but uses
//! random vector embeddings instead of real CLAP for now.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rustfft::{num_complex::Complex, FftPlanner};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use thiserror::Error;
use walkdir::WalkDir;

pub const DEFAULT_TAGS: &[&str] = &[
    "speech",
    "male voice",
    "female voice",
    "music",
    "instrumental",
    "drums",
    "guitar",
    "piano",
    "bass",
    "synth",
    "loop",
    "ambient",
    "crowd",
    "applause",
    "footsteps",
    "rain",
    "wind",
    "birdsong",
    "engine",
    "noise",
];

pub const SUPPORTED_EXTS: &[&str] = &[".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm"];

#[derive(Debug, Error)]
pub enum ClapetyError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Invalid argument: {0}")]
    Invalid(String),
    #[error("Other: {0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, ClapetyError>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    pub top_k: usize,
}
impl Default for InferenceConfig {
    fn default() -> Self {
        Self { top_k: 3 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileCaption {
    pub file: PathBuf,
    pub caption: String,
    pub tags: Vec<String>,
    pub model: String,
}

/// Trait abstraction for an embedding backend (placeholder for real model)
pub trait EmbeddingBackend {
    fn embed_audio(&self, bytes: &[u8]) -> Vec<f32>;
    fn embed_text(&self, texts: &[&str]) -> Vec<Vec<f32>>;
    fn model_name(&self) -> &str;
}

impl EmbeddingBackend for Box<dyn EmbeddingBackend> {
    fn embed_audio(&self, bytes: &[u8]) -> Vec<f32> {
        (**self).embed_audio(bytes)
    }
    fn embed_text(&self, texts: &[&str]) -> Vec<Vec<f32>> {
        (**self).embed_text(texts)
    }
    fn model_name(&self) -> &str {
        (**self).model_name()
    }
}

// ---------------- Audio decoding & resampling ----------------
/// Decode common audio formats to mono f32 PCM with target sample rate.
/// Uses symphonia for multi-format decode. Falls back to empty vec on error.
pub fn decode_and_resample(bytes: &[u8], target_sr: u32) -> (Vec<f32>, u32) {
    use symphonia::core::audio::{AudioBufferRef, Signal, SignalSpec};
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::errors::Error as SymphoniaError;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::default::get_probe;
    let cursor = std::io::Cursor::new(bytes.to_vec());
    let mss = MediaSourceStream::new(Box::new(cursor), Default::default());
    let probed = match get_probe().format(
        &Default::default(),
        mss,
        &FormatOptions::default(),
        &MetadataOptions::default(),
    ) {
        Ok(p) => p,
        Err(_) => return (Vec::new(), target_sr),
    };
    let mut format = probed.format;
    let track = match format.default_track() {
        Some(t) => t.clone(),
        None => return (Vec::new(), target_sr),
    };
    let mut decoder = match symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
    {
        Ok(d) => d,
        Err(_) => return (Vec::new(), target_sr),
    };
    let mut pcm: Vec<f32> = Vec::new();
    let mut src_sr = track.codec_params.sample_rate.unwrap_or(target_sr);
    while let Ok(packet) = format.next_packet() {
        if packet.track_id() != track.id {
            continue;
        }
        match decoder.decode(&packet) {
            Ok(AudioBufferRef::F32(buf)) => {
                src_sr = buf.spec().rate;
                let chans = buf.spec().channels.count();
                for f in 0..buf.frames() {
                    let mut acc = 0.0f32;
                    for c in 0..chans {
                        acc += buf.chan(c)[f];
                    }
                    pcm.push(acc / chans as f32);
                }
            }
            Ok(AudioBufferRef::S16(buf)) => {
                src_sr = buf.spec().rate;
                let chans = buf.spec().channels.count();
                for f in 0..buf.frames() {
                    let mut acc = 0.0f32;
                    for c in 0..chans {
                        acc += buf.chan(c)[f] as f32 / i16::MAX as f32;
                    }
                    pcm.push(acc / chans as f32);
                }
            }
            Ok(AudioBufferRef::U8(buf)) => {
                src_sr = buf.spec().rate;
                let chans = buf.spec().channels.count();
                for f in 0..buf.frames() {
                    let mut acc = 0.0f32;
                    for c in 0..chans {
                        acc += (buf.chan(c)[f] as f32 / 255.0) * 2.0 - 1.0;
                    }
                    pcm.push(acc / chans as f32);
                }
            }
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(_) => break,
            _ => {}
        }
    }
    if src_sr != target_sr && !pcm.is_empty() {
        (linear_resample(&pcm, src_sr, target_sr), target_sr)
    } else {
        (pcm, src_sr)
    }
}

/// Simple linear resampler (mono)
pub fn linear_resample(input: &[f32], src_sr: u32, dst_sr: u32) -> Vec<f32> {
    if src_sr == dst_sr || input.is_empty() {
        return input.to_vec();
    }
    let ratio = dst_sr as f32 / src_sr as f32;
    let out_len = (input.len() as f32 * ratio).ceil() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src_pos = i as f32 / ratio;
        let idx0 = src_pos.floor() as usize;
        if idx0 + 1 >= input.len() {
            out.push(input[idx0]);
            continue;
        }
        let frac = src_pos - idx0 as f32;
        let v = input[idx0] * (1.0 - frac) + input[idx0 + 1] * frac;
        out.push(v);
    }
    out
}

// ---------------- Audio preprocessing (log mel spectrogram) ----------------
// helper to build mel filterbank (placed before use to satisfy compiler)
fn build_mel_filterbank(
    sr: f32,
    n_fft: usize,
    n_mels: usize,
    fmin: f32,
    fmax: f32,
) -> Vec<Vec<f32>> {
    fn hz_to_mel(hz: f32) -> f32 {
        2595.0 * (1.0 + hz / 700.0).log10()
    }
    fn mel_to_hz(m: f32) -> f32 {
        700.0 * (10f32.powf(m / 2595.0) - 1.0)
    }
    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);
    let mel_points: Vec<f32> = (0..(n_mels + 2))
        .map(|i| mel_min + (i as f32) * (mel_max - mel_min) / (n_mels + 1) as f32)
        .collect();
    let hz_points: Vec<f32> = mel_points.iter().map(|m| mel_to_hz(*m)).collect();
    let bin = hz_points
        .iter()
        .map(|hz| ((n_fft + 1) as f32 * hz / sr).floor() as usize)
        .collect::<Vec<_>>();
    let mut fb = vec![vec![0.0; n_fft / 2 + 1]; n_mels];
    for m in 1..(n_mels + 1) {
        let f_m_minus = bin[m - 1];
        let f_m = bin[m];
        let f_m_plus = bin[m + 1];
        for k in f_m_minus..f_m {
            if f_m > f_m_minus {
                fb[m - 1][k] = (k - f_m_minus) as f32 / (f_m - f_m_minus) as f32;
            }
        }
        for k in f_m..f_m_plus {
            if f_m_plus > f_m {
                fb[m - 1][k] = (f_m_plus - k) as f32 / (f_m_plus - f_m) as f32;
            }
        }
    }
    fb
}

pub fn log_mel_spectrogram(pcm: &[f32], sample_rate: u32) -> Vec<f32> {
    let n_fft = 1024usize;
    let hop = (sample_rate as f32 * 0.01) as usize; // 10ms hop
    let win_len = n_fft;
    let n_mels = 64usize; // adjustable
    if pcm.is_empty() {
        return vec![];
    }
    // Hann window
    let window: Vec<f32> = (0..win_len)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / win_len as f32).cos())
        .collect();
    // Mel filterbank
    let mel_fb = build_mel_filterbank(
        sample_rate as f32,
        n_fft,
        n_mels,
        50.0,
        (sample_rate as f32) / 2.0 - 100.0,
    );
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n_fft);
    let mut idx = 0usize;
    let mut out: Vec<f32> = Vec::new();
    while idx + win_len <= pcm.len() {
        let frame = &pcm[idx..idx + win_len];
        let mut buf: Vec<Complex<f32>> = frame
            .iter()
            .zip(window.iter())
            .map(|(s, w)| Complex {
                re: *s * *w,
                im: 0.0,
            })
            .collect();
        buf.resize(n_fft, Complex { re: 0.0, im: 0.0 });
        fft.process(&mut buf);
        let spec_bins = n_fft / 2 + 1;
        let power: Vec<f32> = buf[..spec_bins].iter().map(|c| c.norm_sqr()).collect();
        for row in mel_fb.iter() {
            let mut acc = 0f32;
            for (b, w) in row.iter().enumerate() {
                acc += w * power[b];
            }
            out.push((acc + 1e-10).ln());
        }
        idx += hop;
    }
    out
}

// ---------------- Tokenization wrapper (using tokenizers crate when enabled) -----------
#[cfg(any(feature = "tract", feature = "onnx"))]
pub mod text_tokenizer {
    use super::ClapetyError;
    use std::path::Path;
    use tokenizers::Tokenizer;
    pub struct ClapTokenizer {
        inner: Tokenizer,
    }
    impl ClapTokenizer {
        pub fn from_dir(dir: &Path) -> Result<Self, ClapetyError> {
            let tok_path = dir.join("tokenizer.json");
            let inner =
                Tokenizer::from_file(tok_path).map_err(|e| ClapetyError::Other(e.to_string()))?;
            Ok(Self { inner })
        }
        pub fn encode_batch(
            &self,
            texts: &[&str],
        ) -> Result<(Vec<Vec<u32>>, Vec<Vec<u32>>), ClapetyError> {
            let mut ids = Vec::new();
            let mut masks = Vec::new();
            for t in texts {
                let out = self
                    .inner
                    .encode(*t, true)
                    .map_err(|e| ClapetyError::Other(e.to_string()))?;
                ids.push(out.get_ids().to_vec());
                masks.push(out.get_attention_mask().to_vec());
            }
            Ok((ids, masks))
        }
    }
}

#[cfg(feature = "onnx")]
mod onnx_backend {
    use super::{
        decode_and_resample, l2_normalize, log_mel_spectrogram, text_tokenizer::ClapTokenizer,
        ClapetyError, EmbeddingBackend, Result,
    };
    use lazy_static::lazy_static;
    use onnxruntime::ndarray::{Array2, Array4, Axis};
    use onnxruntime::tensor::OrtOwnedTensor;
    use onnxruntime::Value;
    use onnxruntime::{environment::Environment, session::Session, GraphOptimizationLevel};
    use std::{path::Path, sync::Arc};

    lazy_static! {
        static ref ENV: Arc<Environment> = Arc::new(
            Environment::builder()
                .with_name("clapety-clap")
                .with_log_level(onnxruntime::LoggingLevel::Warning)
                .build()
                .unwrap()
        );
    }

    pub struct OnnxBackend {
        pub audio_session: Session,
        pub text_session: Session,
        pub dim: usize,
        pub name: String,
        pub tokenizer: Option<ClapTokenizer>,
    }

    impl OnnxBackend {
        pub fn from_paths(audio_model: &Path, text_model: &Path) -> Result<Self> {
            let mut session_opts = onnxruntime::SessionOptions::new();
            session_opts.set_graph_optimization_level(GraphOptimizationLevel::Basic); // adjustable
            let audio_session = ENV
                .new_session_builder()?
                .with_session_options(session_opts.clone())
                .with_model_from_file(audio_model)
                .map_err(|e| ClapetyError::Other(e.to_string()))?;
            let text_session = ENV
                .new_session_builder()?
                .with_session_options(session_opts)
                .with_model_from_file(text_model)
                .map_err(|e| ClapetyError::Other(e.to_string()))?;
            // dynamic dim via sibling embedding_dim.txt
            let mut dim: Option<usize> = None;
            let mut tokenizer: Option<ClapTokenizer> = None;
            if let Some(parent) = text_model.parent() {
                let candidate = parent.join("embedding_dim.txt");
                if candidate.exists() {
                    if let Ok(s) = std::fs::read_to_string(&candidate) {
                        dim = s.trim().parse().ok();
                    }
                }
                let tok_dir = parent.join("tokenizer");
                if tok_dir.exists() {
                    if let Ok(tok) = ClapTokenizer::from_dir(&tok_dir) {
                        tokenizer = Some(tok);
                    }
                }
            }
            Ok(Self {
                audio_session,
                text_session,
                dim: dim.unwrap_or(512),
                name: "clap-onnx".into(),
                tokenizer,
            })
        }
    }

    impl EmbeddingBackend for OnnxBackend {
        fn embed_audio(&self, bytes: &[u8]) -> Vec<f32> {
            let target_sr = 48_000u32; // matches export script dummy
            let (pcm, sr) = decode_and_resample(bytes, target_sr);
            if pcm.is_empty() {
                return vec![0.0; self.dim];
            }
            let mel = log_mel_spectrogram(&pcm, sr);
            if mel.is_empty() {
                return vec![0.0; self.dim];
            }
            let n_mels = 64usize; // must align with log_mel_spectrogram
            let frames = mel.len() / n_mels;
            if frames == 0 {
                return vec![0.0; self.dim];
            }
            // Shape: [1,1,frames,n_mels]
            let mut arr = Array4::<f32>::zeros((1, 1, frames, n_mels));
            for f in 0..frames {
                for m in 0..n_mels {
                    arr[[0, 0, f, m]] = mel[f * n_mels + m];
                }
            }
            let input_name = self.audio_session.inputs()[0].name.clone();
            let input_tensor = Value::from_array(self.audio_session.allocator(), &arr).unwrap();
            match self
                .audio_session
                .run(vec![(input_name.as_str(), input_tensor)])
            {
                Ok(mut outputs) => {
                    if let Some(val) = outputs.pop() {
                        if let Ok(tensor) = OrtOwnedTensor::<f32, _>::try_from(val) {
                            let mut v = tensor.iter().cloned().collect::<Vec<f32>>();
                            l2_normalize(&mut v);
                            // If model returns batch x dim; take first row if needed
                            if v.len() > self.dim && v.len() % self.dim == 0 {
                                return v[0..self.dim].to_vec();
                            }
                            return v;
                        }
                    }
                }
                Err(_) => {}
            }
            vec![0.0; self.dim]
        }
        fn embed_text(&self, texts: &[&str]) -> Vec<Vec<f32>> {
            if texts.is_empty() {
                return vec![];
            }
            if let Some(tok) = &self.tokenizer {
                if let Ok((ids, masks)) = tok.encode_batch(texts) {
                    let max_len = ids.iter().map(|v| v.len()).max().unwrap_or(0);
                    let batch = texts.len();
                    let mut ids_arr = Array2::<i64>::zeros((batch, max_len));
                    let mut mask_arr = Array2::<i64>::zeros((batch, max_len));
                    for (i, (row_ids, row_mask)) in ids.iter().zip(masks.iter()).enumerate() {
                        for j in 0..row_ids.len() {
                            ids_arr[[i, j]] = row_ids[j] as i64;
                        }
                        for j in 0..row_mask.len() {
                            mask_arr[[i, j]] = row_mask[j] as i64;
                        }
                    }
                    let in_names = self.text_session.inputs();
                    if in_names.len() >= 2 {
                        let v_ids =
                            Value::from_array(self.text_session.allocator(), &ids_arr).unwrap();
                        let v_mask =
                            Value::from_array(self.text_session.allocator(), &mask_arr).unwrap();
                        let run_res = self.text_session.run(vec![
                            (in_names[0].name.as_str(), v_ids),
                            (in_names[1].name.as_str(), v_mask),
                        ]);
                        if let Ok(mut outs) = run_res {
                            if let Some(val) = outs.pop() {
                                if let Ok(t) = OrtOwnedTensor::<f32, _>::try_from(val) {
                                    // Expect shape [batch, dim]
                                    let view = t.view();
                                    let dim = *view.shape().last().unwrap_or(&self.dim);
                                    let mut all = Vec::new();
                                    for b in 0..batch {
                                        let mut row =
                                            (0..dim).map(|d| view[[b, d]]).collect::<Vec<f32>>();
                                        l2_normalize(&mut row);
                                        all.push(row);
                                    }
                                    return all;
                                }
                            }
                        }
                    }
                }
            }
            // Fallback zeros
            texts.iter().map(|_| vec![0.0; self.dim]).collect()
        }
        fn model_name(&self) -> &str {
            &self.name
        }
    }
    pub use OnnxBackend as Backend;
}

#[cfg(feature = "onnx")]
pub use onnx_backend::Backend as OnnxBackend;

#[cfg(feature = "tract")]
mod tract_backend {
    use super::{
        decode_and_resample, l2_normalize, log_mel_spectrogram, text_tokenizer::ClapTokenizer,
        ClapetyError, EmbeddingBackend, Result,
    };
    use std::path::Path;
    use tract_onnx::prelude::*;

    pub struct TractBackend {
        pub audio_model:
            SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
        pub text_model: SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
        pub dim: usize,
        pub name: String,
        pub tokenizer: Option<ClapTokenizer>,
    }

    impl TractBackend {
        pub fn from_paths(audio: &Path, text: &Path) -> Result<Self> {
            let audio_model = tract_onnx::onnx()
                .model_for_path(audio)
                .map_err(|e| ClapetyError::Other(e.to_string()))?
                .into_optimized()
                .map_err(|e| ClapetyError::Other(e.to_string()))?
                .into_runnable()
                .map_err(|e| ClapetyError::Other(e.to_string()))?;
            let text_model = tract_onnx::onnx()
                .model_for_path(text)
                .map_err(|e| ClapetyError::Other(e.to_string()))?
                .into_optimized()
                .map_err(|e| ClapetyError::Other(e.to_string()))?
                .into_runnable()
                .map_err(|e| ClapetyError::Other(e.to_string()))?;
            // derive dim from model output fact if possible
            let mut dim: Option<usize> = None;
            if let Some(output) = text_model.model().outputs.get(0) {
                if let Ok(fact) = text_model.model().outlet_fact(*output) {
                    if let Some(shape) = fact.shape.clone().as_concrete() {
                        if let Some(last) = shape.last() {
                            dim = Some(*last);
                        }
                    }
                }
            }
            let mut tokenizer: Option<ClapTokenizer> = None;
            if dim.is_none() {
                if let Some(parent) = text.parent() {
                    let cand = parent.join("embedding_dim.txt");
                    if cand.exists() {
                        if let Ok(s) = std::fs::read_to_string(&cand) {
                            dim = s.trim().parse().ok();
                        }
                    }
                    let tok_dir = parent.join("tokenizer");
                    if tok_dir.exists() {
                        if let Ok(tok) = ClapTokenizer::from_dir(&tok_dir) {
                            tokenizer = Some(tok);
                        }
                    }
                }
            }
            Ok(Self {
                audio_model,
                text_model,
                dim: dim.unwrap_or(512),
                name: "clap-tract".into(),
                tokenizer,
            })
        }
    }

    impl EmbeddingBackend for TractBackend {
        fn embed_audio(&self, bytes: &[u8]) -> Vec<f32> {
            let target_sr = 48_000u32;
            let (pcm, sr) = decode_and_resample(bytes, target_sr);
            if pcm.is_empty() {
                return vec![0.0; self.dim];
            }
            let mel = log_mel_spectrogram(&pcm, sr);
            if mel.is_empty() {
                return vec![0.0; self.dim];
            }
            let n_mels = 64usize;
            let frames = mel.len() / n_mels;
            if frames == 0 {
                return vec![0.0; self.dim];
            }
            // Tensor shape assumptions: [1,1,frames,n_mels]
            let mut tensor = tract_ndarray::Array4::<f32>::zeros((1, 1, frames, n_mels));
            for f in 0..frames {
                for m in 0..n_mels {
                    tensor[[0, 0, f, m]] = mel[f * n_mels + m];
                }
            }
            let input = tract_onnx::prelude::Tensor::from(tensor);
            match self.audio_model.run(tvec!(input.into())) {
                Ok(outputs) => {
                    if let Some(t) = outputs.get(0) {
                        if let Ok(slice) = t.to_array_view::<f32>() {
                            let mut v = slice.iter().cloned().collect::<Vec<f32>>();
                            l2_normalize(&mut v);
                            if v.len() > self.dim && v.len() % self.dim == 0 {
                                return v[0..self.dim].to_vec();
                            }
                            return v;
                        }
                    }
                }
                Err(_) => {}
            }
            vec![0.0; self.dim]
        }
        fn embed_text(&self, texts: &[&str]) -> Vec<Vec<f32>> {
            if texts.is_empty() {
                return vec![];
            }
            if let Some(tok) = &self.tokenizer {
                if let Ok((ids, masks)) = tok.encode_batch(texts) {
                    let max_len = ids.iter().map(|v| v.len()).max().unwrap_or(0);
                    let batch = texts.len();
                    let mut ids_arr = tract_ndarray::Array2::<i64>::zeros((batch, max_len));
                    let mut mask_arr = tract_ndarray::Array2::<i64>::zeros((batch, max_len));
                    for (i, (row_ids, row_mask)) in ids.iter().zip(masks.iter()).enumerate() {
                        for j in 0..row_ids.len() {
                            ids_arr[[i, j]] = row_ids[j] as i64;
                        }
                        for j in 0..row_mask.len() {
                            mask_arr[[i, j]] = row_mask[j] as i64;
                        }
                    }
                    let ids_tensor = tract_onnx::prelude::Tensor::from(ids_arr);
                    let mask_tensor = tract_onnx::prelude::Tensor::from(mask_arr);
                    if let Ok(outputs) = self
                        .text_model
                        .run(tvec!(ids_tensor.into(), mask_tensor.into()))
                    {
                        if let Some(t) = outputs.get(0) {
                            if let Ok(view) = t.to_array_view::<f32>() {
                                let dim = *view.shape().last().unwrap_or(&self.dim);
                                let mut out = Vec::new();
                                for b in 0..batch {
                                    let mut row =
                                        (0..dim).map(|d| view[[b, d]]).collect::<Vec<f32>>();
                                    l2_normalize(&mut row);
                                    out.push(row);
                                }
                                return out;
                            }
                        }
                    }
                }
            }
            texts.iter().map(|_| vec![0.0; self.dim]).collect()
        }
        fn model_name(&self) -> &str {
            &self.name
        }
    }

    pub use TractBackend as Backend;
}

#[cfg(feature = "tract")]
pub use tract_backend::Backend as TractBackend;

/// Dummy backend: random but deterministic for session.
#[derive(Clone)]
pub struct DummyBackend {
    rng: StdRng,
    dim: usize,
    name: String,
}
impl DummyBackend {
    pub fn new() -> Self {
        Self {
            rng: StdRng::seed_from_u64(42),
            dim: 64,
            name: "dummy-clap".into(),
        }
    }
}
impl EmbeddingBackend for DummyBackend {
    fn embed_audio(&self, _bytes: &[u8]) -> Vec<f32> {
        (0..self.dim).map(|i| (i as f32).sin()).collect()
    }
    fn embed_text(&self, texts: &[&str]) -> Vec<Vec<f32>> {
        texts
            .iter()
            .enumerate()
            .map(|(i, _t)| (0..self.dim).map(|j| ((i + j) as f32).cos()).collect())
            .collect()
    }
    fn model_name(&self) -> &str {
        &self.name
    }
}

pub struct TagInferencer<B: EmbeddingBackend> {
    backend: B,
}
impl<B: EmbeddingBackend> TagInferencer<B> {
    pub fn new(backend: B) -> Self {
        Self { backend }
    }
    pub fn infer_paths(
        &self,
        paths: Vec<PathBuf>,
        cfg: &InferenceConfig,
    ) -> Result<Vec<FileCaption>> {
        let files = expand_audio_files(paths)?;
        files
            .into_iter()
            .map(|f| self.infer_file(&f, cfg))
            .collect()
    }
    pub fn infer_file(&self, file: &Path, cfg: &InferenceConfig) -> Result<FileCaption> {
        let mut buf = Vec::new();
        File::open(file)?.read_to_end(&mut buf)?;
        let audio_emb = self.backend.embed_audio(&buf);
        let text_embs = self.backend.embed_text(DEFAULT_TAGS);
        let mut sims: Vec<(usize, f32)> = text_embs
            .iter()
            .enumerate()
            .map(|(i, e)| (i, cosine(&audio_emb, e)))
            .collect();
        sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top_k = cfg.top_k.min(sims.len());
        let tags: Vec<String> = sims
            .into_iter()
            .take(top_k)
            .map(|(i, _)| DEFAULT_TAGS[i].to_string())
            .collect();
        let caption = tags.join(", ");
        Ok(FileCaption {
            file: file.to_path_buf(),
            caption,
            tags,
            model: self.backend.model_name().to_string(),
        })
    }
}

impl Default for TagInferencer<Box<dyn EmbeddingBackend>> {
    fn default() -> Self {
        TagInferencer::new(Box::new(DummyBackend::new()) as Box<dyn EmbeddingBackend>)
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Recursively expand audio files from provided paths
pub fn expand_audio_files(inputs: Vec<PathBuf>) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for p in inputs {
        if p.is_file() {
            if is_supported(&p) {
                out.push(p);
            }
        } else if p.is_dir() {
            for entry in WalkDir::new(&p) {
                let e = entry.map_err(|e| ClapetyError::Other(e.to_string()))?;
                if e.file_type().is_file() {
                    let pb = e.path().to_path_buf();
                    if is_supported(&pb) {
                        out.push(pb);
                    }
                }
            }
        } else {
            return Err(ClapetyError::Invalid(format!(
                "Path not found: {}",
                p.display()
            )));
        }
    }
    Ok(out)
}

fn is_supported(p: &Path) -> bool {
    p.extension()
        .and_then(|s| s.to_str())
        .map(|ext| {
            let ext = format!(".{}", ext).to_lowercase();
            SUPPORTED_EXTS.contains(&ext.as_str())
        })
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn smoke() {
        let infer = TagInferencer::default();
        let cfg = InferenceConfig { top_k: 3 };
        // won't actually read audio; if file missing this test is trivial to adjust
        let fake = PathBuf::from("Cargo.toml"); // unsupported ext -> expand_audio_files would filter; we call infer_file directly
        let res = infer.backend.embed_audio(&[]);
        assert_eq!(res.len(), 64);
        let tags_embs = infer.backend.embed_text(DEFAULT_TAGS);
        assert_eq!(tags_embs.len(), DEFAULT_TAGS.len());
        let _ = cosine(&tags_embs[0], &tags_embs[1]);
    }
}
