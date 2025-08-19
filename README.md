## CLAP Audio Captioning (CLI & Web)

Tools (CLI + FastAPI web UI) that use the CLAP model to embed audio and rank a fixed vocabulary of semantic tags, producing a comma‑separated pseudo caption (top‑K tags).

### Install / Sync (uv)

This project is managed with [uv](https://github.com/astral-sh/uv); you don't need to create or activate a virtual environment manually.

1. Install uv (one time):
   - macOS (Homebrew): `brew install uv`
   - Or via script: `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. Resolve & install dependencies (creates an isolated environment automatically):

```
uv sync
```

3. Run the CLI (from the project root):

```
uv run clap caption path/to/audio.wav
```

If you ever see `Failed to spawn: clap` after changing packaging config, re-run `uv sync`. As a fallback you can also invoke via module:

```
uv run python -m clap.cli caption path/to/audio.wav
```

Editable development (auto-reload your local package) is already handled by uv using the workspace source; no extra `-e` flag is required.

### Usage

Caption a single file (prints JSON to stdout, table to stderr):

```
uv run clap caption path/to/audio.wav
```

Caption multiple files / directories, output JSONL:

```
uv run clap caption samples/ --output captions.jsonl
```

Write individual caption text files:

```
uv run clap caption samples/ --output out_dir/
```

Disable table and just get JSON to stdout:

```
uv run clap caption samples/*.wav --no-table
```

### Notes

Captions are the top‑K ranked tags (default 3) chosen from a default vocabulary (`DEFAULT_TAGS` in `clap/core.py`). Adjust K with `--top-k`. For natural language sentence generation you'd use a generative captioning model; CLAP provides contrastive embeddings for ranking.

### Web Server

Start server (serves API + static UI):

```
uv run clap-server
```

Open http://127.0.0.1:8000 and drag & drop an audio file. Inference starts automatically.

API endpoint:

```
POST /api/caption
Form: file=<audio>, top_k (int, optional), model_name (str, optional)
Returns JSON: {"caption", "tags": [...], "model", "filename", "top_k"}
```

Example:

```
curl -F file=@tests/fixtures/loop.wav -F top_k=5 http://127.0.0.1:8000/api/caption
```
