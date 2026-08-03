from pathlib import Path

import torch

from vllm.model_executor.models.apertus_utils import (
    ApertusAudioTokenizer,
    ApertusImageTokenizer,
)


def test_image_tokenizer_accepts_published_filename(
    tmp_path: Path, monkeypatch,
) -> None:
    (tmp_path / "model-vision_tokenizer-model.safetensors").touch()
    class FakeVisionTokenizer:
        def to(self, *, dtype: torch.dtype) -> "FakeVisionTokenizer":
            assert dtype == torch.float32
            return self

    expected_tokenizer = FakeVisionTokenizer()
    monkeypatch.setattr(
        "vllm.model_executor.models.apertus_emu35.build_vision_tokenizer",
        lambda **kwargs: expected_tokenizer,
    )

    tokenizer = ApertusImageTokenizer().load_vision_tokenizer(
        model_path=str(tmp_path),
        device="cuda:0",
        dtype=torch.float32,
        vision_config={},
    )

    assert tokenizer is expected_tokenizer


def test_audio_tokenizer_accepts_published_filename(
    tmp_path: Path, monkeypatch,
) -> None:
    (tmp_path / "model-wavtokenizer-model.safetensors").touch()
    expected_tokenizer = object()
    monkeypatch.setattr(
        "vllm.model_executor.models.apertus_wavetokenizer.build_audio_tokenizer",
        lambda **kwargs: expected_tokenizer,
    )

    tokenizer = ApertusAudioTokenizer().load_audio_tokenizer(
        model_path=str(tmp_path),
        device="cuda:0",
        audio_config={},
    )

    assert tokenizer is expected_tokenizer
