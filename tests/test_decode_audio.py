import io
import struct
import wave
from subprocess import CompletedProcess
from unittest.mock import patch

import numpy as np
import pytest
from fastapi import HTTPException

from app.routers.stt import _decode_audio


def _make_wav_bytes(samples: list[int]) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return buf.getvalue()


def test_decode_audio_returns_float32_array():
    wav_bytes = _make_wav_bytes([0, 16384, -16384, 8192])
    mock_result = CompletedProcess(args=[], returncode=0, stdout=wav_bytes, stderr=b"")

    with patch("app.routers.stt.subprocess.run", return_value=mock_result):
        result = _decode_audio(b"fake_audio_bytes")

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert len(result) == 4
    assert pytest.approx(result[1]) == 16384 / (2**15)


def test_decode_audio_invalid_bytes_raises_http_400():
    mock_result = CompletedProcess(args=[], returncode=1, stdout=b"", stderr=b"fichier corrompu")

    with patch("app.routers.stt.subprocess.run", return_value=mock_result):
        with pytest.raises(HTTPException) as exc_info:
            _decode_audio(b"bytes invalides")

    assert exc_info.value.status_code == 400
    assert "Format audio invalide" in exc_info.value.detail
