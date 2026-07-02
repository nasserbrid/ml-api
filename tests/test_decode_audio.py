from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi import HTTPException

from app.routers.stt import _decode_audio


def test_decode_audio_returns_float32_array():
    final_segment = MagicMock()
    final_segment.get_array_of_samples.return_value = [0, 16384, -16384, 8192]

    with patch("app.routers.stt.AudioSegment") as mock_audio:
        mock_audio.from_file.return_value.set_channels.return_value.set_frame_rate.return_value = final_segment

        result = _decode_audio(b"fake_audio_bytes")

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert len(result) == 4
    assert pytest.approx(result[1]) == 16384 / (2**15)


def test_decode_audio_invalid_bytes_raises_http_400():
    with patch("app.routers.stt.AudioSegment") as mock_audio:
        mock_audio.from_file.side_effect = Exception("fichier corrompu")

        with pytest.raises(HTTPException) as exc_info:
            _decode_audio(b"bytes invalides")

    assert exc_info.value.status_code == 400
    assert "Format audio invalide" in exc_info.value.detail
