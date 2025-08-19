use clap::Parser;
use clapety_clap::{DummyBackend, EmbeddingBackend, InferenceConfig, TagInferencer};
use colored::*;
use std::io::{self, Write};
use serde_json;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "clapety-clap",
    about = "Rust audio tag ranking (placeholder backend)"
)]
struct Args {
    /// Files or directories
    #[arg(required = true)]
    paths: Vec<PathBuf>,

    /// Top-K tags
    #[arg(long = "top-k", default_value_t = 3)]
    top_k: usize,

    /// Output JSON (array) to stdout only
    #[arg(long = "json", default_value_t = false)]
    json: bool,

    /// Show pretty table (stderr); disable with --no-table
    #[arg(long = "show-table", default_value_t = true, action=clap::ArgAction::Set )]
    show_table: bool,

    /// Backend: dummy | onnx (requires --features onnx)
    #[arg(long = "backend", default_value = "dummy")]
    backend: String,

    /// Path to audio encoder ONNX (when backend=onnx)
    #[arg(long = "audio-model", requires = "backend")]
    audio_model: Option<PathBuf>,

    /// Path to text encoder ONNX (when backend=onnx)
    #[arg(long = "text-model", requires = "backend")]
    text_model: Option<PathBuf>,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let boxed_backend: Box<dyn EmbeddingBackend> = match args.backend.as_str() {
        #[cfg(feature = "onnx")]
        "onnx" => {
            use clapety_clap::OnnxBackend;
            let a = args
                .audio_model
                .as_ref()
                .expect("--audio-model required for onnx");
            let t = args
                .text_model
                .as_ref()
                .expect("--text-model required for onnx");
            Box::new(OnnxBackend::from_paths(a, t).expect("load onnx models"))
        }
        #[cfg(feature = "tract")]
        "tract" => {
            use clapety_clap::TractBackend;
            let a = args
                .audio_model
                .as_ref()
                .expect("--audio-model required for tract");
            let t = args
                .text_model
                .as_ref()
                .expect("--text-model required for tract");
            Box::new(TractBackend::from_paths(a, t).expect("load tract models"))
        }
        _ => Box::new(DummyBackend::new()),
    };
    let infer = TagInferencer::new(boxed_backend);
    let cfg = InferenceConfig { top_k: args.top_k };
    let results = infer.infer_paths(args.paths.clone(), &cfg)?;

    // Table to stderr if requested
    if args.show_table && !args.json {
        let mut table = comfy_table::Table::new();
        table.set_header(vec!["File".bold(), "Caption (top-k tags)".bold()]);
        for r in &results {
            let fname = r.file.file_name().and_then(|s| s.to_str()).unwrap_or("?");
            table.add_row(vec![fname.to_string(), r.caption.clone()]);
        }
        eprintln!("{}", "Captions".green().bold());
        eprintln!("{}", table);
    }
    // JSON to stdout (pretty unless user wants terse later)
    println!("{}", serde_json::to_string_pretty(&results)?);
    Ok(())
}
