import io
import os
import struct
import wave
from subprocess import CompletedProcess
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from app.routers.stt import _normalize_to_wav


def _make_wav_bytes(samples: list[int]) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return buf.getvalue()


def test_normalize_to_wav_returns_path_to_valid_wav():
    wav_bytes = _make_wav_bytes([0, 16384, -16384, 8192])

    def fake_ffmpeg_run(cmd, capture_output=True):
        output_path = cmd[-1]
        with open(output_path, "wb") as f:
            f.write(wav_bytes)
        return CompletedProcess(args=cmd, returncode=0, stdout=b"", stderr=b"")

    with patch("app.routers.stt.subprocess.run", side_effect=fake_ffmpeg_run):
        wav_path = _normalize_to_wav(b"fake_audio_bytes")

    try:
        assert os.path.exists(wav_path)
        with open(wav_path, "rb") as f:
            assert f.read() == wav_bytes
    finally:
        os.unlink(wav_path)


def test_normalize_to_wav_invalid_bytes_raises_http_400():
    def fake_ffmpeg_run(cmd, capture_output=True):
        return CompletedProcess(args=cmd, returncode=1, stdout=b"", stderr=b"fichier corrompu")

    with patch("app.routers.stt.subprocess.run", side_effect=fake_ffmpeg_run):
        with pytest.raises(HTTPException) as exc_info:
            _normalize_to_wav(b"bytes invalides")

    assert exc_info.value.status_code == 400
    assert "Format audio invalide" in exc_info.value.detail
