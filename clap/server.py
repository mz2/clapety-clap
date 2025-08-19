"""FastAPI server exposing CLAP tag inference plus simple web UI."""
from __future__ import annotations

import os
import pathlib
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .core import (
    load_model,
    compute_clap_tags,
    DEFAULT_TAGS,
    MODEL_REPO_DEFAULT,
    build_caption,
)

APP_TITLE = "CLAP Tag Inference"

app = FastAPI(title=APP_TITLE)

STATIC_DIR = pathlib.Path(__file__).parent / "static"
if STATIC_DIR.exists():  # mount if created
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def get_model(model_name: str):  # cached via lru in core
    processor, model = load_model(model_name)
    return processor, model


@app.get("/", response_class=HTMLResponse)
def index():  # pragma: no cover - static content
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return index_path.read_text(encoding="utf-8")
    return "<h1>CLAP Server</h1><p>Static assets missing.</p>"


@app.post("/api/caption")
async def caption_api(
    file: UploadFile = File(...),
    top_k: int = Form(3),
    model_name: str = Form(MODEL_REPO_DEFAULT),
):
    if top_k <= 0 or top_k > 50:
        raise HTTPException(status_code=400, detail="top_k must be 1..50")
    suffix = pathlib.Path(file.filename or "").suffix.lower()
    if suffix not in {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm"}:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    processor, model = get_model(model_name)
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")
    tmp_path = pathlib.Path(f"/tmp/clap_upload_{os.getpid()}_{file.filename}")
    with tmp_path.open("wb") as f:
        f.write(raw)

    tag_list: List[str] = DEFAULT_TAGS

    try:
        ranked = compute_clap_tags(tmp_path, processor, model, tag_list, top_k=top_k)
        caption = build_caption(ranked)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

    return JSONResponse(
        {
            "filename": file.filename,
            "caption": caption,
            "tags": ranked,
            "model": model.config.name_or_path,
            "top_k": top_k,
        }
    )


def main():  # pragma: no cover
    import uvicorn

    uvicorn.run("clap.server:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":  # pragma: no cover
    main()
