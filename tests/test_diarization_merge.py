from types import SimpleNamespace

from app.routers.stt import _merge


class _FakeDiarization:
    def __init__(self, turns: list[tuple[float, float, str]]):
        self._turns = turns

    def itertracks(self, yield_label: bool = True):
        for start, end, label in self._turns:
            yield SimpleNamespace(start=start, end=end), None, label


def test_merge_assigns_speaker_by_segment_midpoint():
    diarization = _FakeDiarization([(0.0, 2.0, "SPEAKER_00"), (2.0, 4.0, "SPEAKER_01")])
    fw_segments = [
        SimpleNamespace(start=0.0, end=1.5, text=" Bonjour "),
        SimpleNamespace(start=2.5, end=3.5, text=" merci "),
    ]

    merged = _merge(diarization, fw_segments)

    assert merged == [
        {"speaker": "SPEAKER_00", "text": "Bonjour", "start": 0.0, "end": 1.5},
        {"speaker": "SPEAKER_01", "text": "merci", "start": 2.5, "end": 3.5},
    ]


def test_merge_concatenates_consecutive_segments_from_same_speaker():
    diarization = _FakeDiarization([(0.0, 5.0, "SPEAKER_00")])
    fw_segments = [
        SimpleNamespace(start=0.0, end=1.0, text="Bonjour"),
        SimpleNamespace(start=1.0, end=2.0, text="tout le monde"),
    ]

    merged = _merge(diarization, fw_segments)

    assert merged == [
        {"speaker": "SPEAKER_00", "text": "Bonjour tout le monde", "start": 0.0, "end": 2.0}
    ]


def test_merge_defaults_to_speaker_00_when_no_matching_turn():
    diarization = _FakeDiarization([])
    fw_segments = [SimpleNamespace(start=0.0, end=1.0, text="Bonjour")]

    merged = _merge(diarization, fw_segments)

    assert merged == [{"speaker": "SPEAKER_00", "text": "Bonjour", "start": 0.0, "end": 1.0}]
