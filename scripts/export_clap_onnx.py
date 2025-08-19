#!/usr/bin/env python
"""Export CLAP (laion/clap-htsat-fused) audio & text encoders to ONNX.

Produces two models:
  audio_encoder.onnx  (expects float32 mel or raw waveform? -> we export with waveform input here)
  text_encoder.onnx   (expects input_ids + attention_mask)

NOTE: For a production-quality export you'd freeze preprocessing separately
(mel spectrogram) and feed that to the audio encoder. For now we keep the
processor's waveform path and rely on Rust to replicate preprocessing later.
"""
from __future__ import annotations
import argparse
import pathlib
import torch
from transformers import ClapModel, ClapProcessor

DEFAULT_MODEL = "laion/clap-htsat-fused"


def export(model_name: str, out_dir: pathlib.Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    processor = ClapProcessor.from_pretrained(model_name)
    model = ClapModel.from_pretrained(model_name)
    model.eval()

    # Audio path export (waveform input) ---------------------------------
    # Create dummy 1 second 48k waveform
    waveform = torch.zeros(48000, dtype=torch.float32)
    audio_inputs = processor(
        audios=waveform.numpy(), sampling_rate=48000, return_tensors="pt"
    )
    # Identify expected input names
    audio_input_tensors = tuple(
        audio_inputs[k]
        for k in audio_inputs
        if isinstance(audio_inputs[k], torch.Tensor)
    )
    audio_input_names = [
        k for k in audio_inputs if isinstance(audio_inputs[k], torch.Tensor)
    ]
    audio_out_path = out_dir / "audio_encoder.onnx"
    with torch.no_grad():
        torch.onnx.export(
            model,
            audio_input_tensors,
            str(audio_out_path),
            input_names=audio_input_names,
            output_names=["audio_emb"],
            opset_version=17,
            do_constant_folding=True,
            dynamic_axes={n: {0: "batch"} for n in audio_input_names}
            | {"audio_emb": {0: "batch"}},
        )
    print(f"Exported audio encoder -> {audio_out_path}")

    # Text path export ---------------------------------------------------
    sample_tags = ["music", "speech"]
    text_inputs = processor(text=sample_tags, return_tensors="pt", padding=True)
    text_input_tensors = tuple(
        text_inputs[k] for k in text_inputs if isinstance(text_inputs[k], torch.Tensor)
    )
    text_input_names = [
        k for k in text_inputs if isinstance(text_inputs[k], torch.Tensor)
    ]
    text_out_path = out_dir / "text_encoder.onnx"

    class TextWrapper(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base

        def forward(self, input_ids, attention_mask):  # type: ignore
            return self.base.get_text_features(
                input_ids=input_ids, attention_mask=attention_mask
            )

    text_module = TextWrapper(model)
    with torch.no_grad():
        torch.onnx.export(
            text_module,
            text_input_tensors,
            str(text_out_path),
            input_names=text_input_names,
            output_names=["text_emb"],
            opset_version=17,
            do_constant_folding=True,
            dynamic_axes={n: {0: "batch"} for n in text_input_names}
            | {"text_emb": {0: "batch"}},
        )
    print(f"Exported text encoder -> {text_out_path}")

    # Save tokenizer artifacts for Rust tokenizers crate consumption
    tok_dir = out_dir / "tokenizer"
    tok_dir.mkdir(exist_ok=True)
    processor.tokenizer.save_pretrained(tok_dir)
    print(f"Saved tokenizer to {tok_dir}")

    # Embedding dimension (derive from text_emb shape)
    with torch.no_grad():
        emb = text_module(*text_input_tensors)
    dim = emb.shape[-1]
    (out_dir / "embedding_dim.txt").write_text(str(dim))
    print(f"Embedding dim: {dim}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("onnx-export"))
    args = ap.parse_args()
    export(args.model, args.out)
