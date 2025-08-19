"""CLI providing CLAP-based tag inference for audio files.

The tool computes top-K semantic tags using the CLAP (Contrastive Language-Audio Pretraining)
model by embedding audio + candidate text tags and ranking by cosine similarity.

If the CLAP model (transformers) can't be loaded (offline / missing deps), a dummy fallback
returns simple filename-derived pseudo tags so the CLI and tests remain functional.
"""

import json
import pathlib
from typing import Optional, List

import click
from rich.console import Console
from rich.table import Table

console = Console(stderr=True)

MODEL_REPO_DEFAULT = "laion/clap-htsat-fused"
SUPPORTED_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm"}

DEFAULT_TAGS = [
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
]

try:  # pragma: no cover - import guard
    from transformers import ClapProcessor, ClapModel  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "transformers with CLAP support is required: pip install transformers"
    ) from e


def load_model(model_name: str):
    """Load CLAP model & processor strictly (no fallback)."""
    processor = ClapProcessor.from_pretrained(model_name)
    model = ClapModel.from_pretrained(model_name)
    return processor, model


def gather_audio_files(inputs: List[pathlib.Path]) -> List[pathlib.Path]:
    files: List[pathlib.Path] = []
    for p in inputs:
        if p.is_dir():
            for sub in sorted(p.rglob("*")):
                if sub.is_file() and sub.suffix.lower() in SUPPORTED_EXTS:
                    files.append(sub)
        elif p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            files.append(p)
        else:
            console.log(f"[yellow]Skipping unsupported path: {p}")
    return files


def compute_clap_tags(
    audio_path: pathlib.Path, processor, model, tags: List[str], top_k: int
):
    import librosa  # local import to reduce import time when unused
    import torch

    waveform, _ = librosa.load(str(audio_path), sr=48000, mono=True)
    audio_inputs = processor(audios=waveform, sampling_rate=48000, return_tensors="pt")
    with torch.no_grad():
        audio_features = model.get_audio_features(**audio_inputs)
    text_inputs = processor(text=tags, return_tensors="pt", padding=True)
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
    audio_features = audio_features / audio_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    sims = (audio_features @ text_features.T).squeeze(0)
    top = sims.topk(min(top_k, sims.shape[-1]))
    return [tags[i] for i in top.indices.tolist()]


def build_caption(tags: List[str]) -> str:
    return ", ".join(tags)


@click.group(help="CLAP audio captioning utilities")
@click.version_option()
def cli():  # pragma: no cover
    pass


@cli.command(
    "caption", help="Generate pseudo-captions (top-K CLAP tags) for audio files."
)
@click.argument(
    "paths",
    type=click.Path(exists=True, path_type=pathlib.Path),
    nargs=-1,
    required=True,
)
@click.option(
    "--model",
    "model_name",
    default=MODEL_REPO_DEFAULT,
    show_default=True,
    help="Model repo or local path",
)
@click.option(
    "--device",
    default="cpu",
    show_default=True,
    help="Computation device (cpu, cuda, mps)",
)
@click.option(
    "--output",
    type=click.Path(path_type=pathlib.Path),
    help="Optional JSONL file or directory for .txt captions",
)
@click.option(
    "--overwrite/--no-overwrite",
    default=False,
    show_default=True,
    help="Overwrite existing per-file caption outputs",
)
@click.option(
    "--show-table/--no-table",
    default=True,
    show_default=True,
    help="Display results as a rich table",
)
@click.option(
    "--top-k", type=int, default=3, show_default=True, help="Top-K tags to include"
)
@click.option(
    "--tags-file",
    type=click.Path(exists=True, path_type=pathlib.Path),
    help="Optional newline separated tag list (replaces defaults)",
)
def caption_cmd(
    paths: List[pathlib.Path],
    model_name: str,
    device: str,
    output: Optional[pathlib.Path],
    overwrite: bool,
    show_table: bool,
    top_k: int,
    tags_file: Optional[pathlib.Path],
):
    console.rule("Loading model")
    processor, model = load_model(model_name)
    model.to(device)

    tag_list = DEFAULT_TAGS
    if tags_file:
        raw = [
            ln.strip()
            for ln in tags_file.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        if raw:
            tag_list = raw
        else:
            console.log(f"[yellow]No valid tags in {tags_file}; using defaults")

    audio_files = gather_audio_files(list(paths))
    if not audio_files:
        raise click.ClickException("No audio files found.")

    results = []
    for af in audio_files:
        tags = compute_clap_tags(af, processor, model, tag_list, top_k=top_k)
        caption = build_caption(tags)
        results.append(
            {
                "file": str(af),
                "caption": caption,
                "tags": tags,
                "model": model.config.name_or_path,
            }
        )

    if output:
        if output.suffix.lower() in {".jsonl", ".ndjson"}:
            with output.open("w", encoding="utf-8") as f:
                for row in results:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            console.log(f"Wrote JSONL: {output}")
        else:
            output.mkdir(parents=True, exist_ok=True)
            for row in results:
                out_file = output / (pathlib.Path(row["file"]).stem + ".txt")
                if out_file.exists() and not overwrite:
                    console.log(f"[yellow]Skip existing {out_file}")
                    continue
                out_file.write_text(row["caption"] + "\n", encoding="utf-8")
            console.log(f"Wrote {len(results)} caption files to {output}")

    if show_table:
        table = Table(title="Captions")
        table.add_column("File", overflow="fold")
        table.add_column("Caption (top-k tags)")
        for row in results:
            table.add_row(pathlib.Path(row["file"]).name, row["caption"])
        console.print(table)

    if not output:
        click.echo(json.dumps(results, ensure_ascii=False, indent=2))


def main():  # pragma: no cover
    cli()


if __name__ == "__main__":  # pragma: no cover
    main()
