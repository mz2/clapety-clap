import json
import subprocess
import sys
from pathlib import Path


def test_cli_smoke(tmp_path: Path):
    # create dummy wav file (silence) using soundfile
    import numpy as np
    import soundfile as sf

    sr = 16000
    data = np.zeros(sr, dtype="float32")
    audio_path = tmp_path / "silence.wav"
    sf.write(audio_path, data, sr)

    cmd = [sys.executable, "-m", "clap.cli", "caption", str(audio_path), "--no-table"]
    out = subprocess.check_output(cmd)
    payload = json.loads(out)
    assert payload and payload[0]["file"].endswith("silence.wav")
    # Caption should be a comma-separated list of 1+ tags
    caption = payload[0]["caption"]
    assert isinstance(caption, str) and "," in caption or len(caption.split()) >= 1
    assert "model" in payload[0]
