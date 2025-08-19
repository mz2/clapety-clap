## CLAP Audio Captioning CLI

Prototype CLI wrapping a (placeholder) CLAP-like audio captioning model.

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

Editable development (auto-reload your local package) is already handled by uv using the workspace source; no extra `-e` flag is required.

### Usage

Caption a single file:

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

Current caption generation is a placeholder that echoes the filename. Replace `caption_audio_file` in `clap/cli.py` with real CLAP model inference logic (feature extraction + generation / decoding).
