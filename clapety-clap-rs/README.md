# clapety-clap (Rust)

Rust library + CLI mirroring the Python clapety-clap tool. Provides:

- Library crate (`clapety_clap`) exposing a simple audio tag ranking API
- Binary `clapety-clap` for command-line use
- Designed so the core logic is `no_std` friendly (with minor gate adjustments) to target WebAssembly next.

> NOTE: This initial version uses placeholder random embeddings instead of real CLAP model inference. It preserves the interface so a subsequent step can integrate an actual model backend (e.g. candle + converted weights, ONNX Runtime, or a smaller distilled variant). This keeps the scaffolding ready for WASM.

## Goals

1. Mirror Python CLI flags: paths, top-k, JSON output, table-like stdout.
2. Provide lib functions for: collecting audio files, computing (mock) tags, building caption.
3. Keep data structures serializable via `serde`.
4. Make core independent from specific model backend; insert trait abstraction.

## Quick Start

```bash
cd clapety-clap-rs
cargo run -- --help
```

Run on a directory of audio:

```bash
cargo run -- samples/ --top-k 5 --json
```

## Library Usage

```rust
use clapety_clap::{TagInferencer, InferenceConfig};

let infer = TagInferencer::default();
let cfg = InferenceConfig { top_k: 3 };
let res = infer.infer_paths(vec!["tests/data/silence.wav".into()], &cfg)?;
println!("{:#?}", res);
```

## Future Work

- Integrate real CLAP embeddings (likely via candle or ONNX)
- (In progress) ONNX backend scaffolding behind `--features onnx` with `--backend onnx --audio-model path --text-model path`
- Tract (for WASM viability) backend scaffold behind `--features tract` with `--backend tract --audio-model path --text-model path`
- WASM target: feature `wasm` to remove filesystem walking / use web audio buffers
- Add deterministic seed and proper similarity scoring
- Streaming / chunked audio support

## License

Dual-licensed under MIT or Apache-2.0.
