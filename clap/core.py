"""Core CLAP tag inference utilities shared by CLI and web server."""

from __future__ import annotations

import pathlib
from functools import lru_cache
from typing import List

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

try:  # pragma: no cover
    from transformers import ClapProcessor, ClapModel  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "transformers with CLAP support is required: pip install transformers"
    ) from e


@lru_cache(maxsize=2)
def load_model(model_name: str = MODEL_REPO_DEFAULT):  # pragma: no cover (heavy path)
    processor = ClapProcessor.from_pretrained(model_name)
    model = ClapModel.from_pretrained(model_name)
    return processor, model


def compute_clap_tags(
    audio_path: pathlib.Path, processor, model, tags: List[str], top_k: int
) -> List[str]:
    import librosa  # defer heavy deps
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


def gather_audio_files(inputs: List[pathlib.Path]) -> List[pathlib.Path]:
    files: List[pathlib.Path] = []
    for p in inputs:
        if p.is_dir():
            for sub in sorted(p.rglob("*")):
                if sub.is_file() and sub.suffix.lower() in SUPPORTED_EXTS:
                    files.append(sub)
        elif p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            files.append(p)
    return files
