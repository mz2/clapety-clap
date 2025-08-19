"""CLI providing CLAP-based tag inference for audio files."""

import json
import pathlib
from typing import Optional, List

import click
from rich.console import Console
from rich.table import Table

from .core import (
    MODEL_REPO_DEFAULT,
    DEFAULT_TAGS,
    gather_audio_files,
    load_model,
    compute_clap_tags,
    build_caption,
)

console = Console(stderr=True)


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
def caption_cmd(
    paths: List[pathlib.Path],
    model_name: str,
    device: str,
    output: Optional[pathlib.Path],
    overwrite: bool,
    show_table: bool,
    top_k: int,
):
    console.rule("Loading model")
    processor, model = load_model(model_name)
    model.to(device)

    tag_list = DEFAULT_TAGS

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
