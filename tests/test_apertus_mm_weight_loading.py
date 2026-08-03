from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.model_executor.models.apertus_mm import Apertus1p5ForConditionalGeneration


def test_generic_loader_skips_towers_loaded_from_dedicated_files() -> None:
    model = Apertus1p5ForConditionalGeneration.__new__(
        Apertus1p5ForConditionalGeneration
    )
    model.config = SimpleNamespace(tie_word_embeddings=False)
    model.vision_tower = object()
    model.audio_tower = object()
    model._model_path = "/model"

    captured_skip_prefixes: list[str] = []

    class CapturingWeightsLoader:
        def __init__(self, module: object, *, skip_prefixes: list[str]) -> None:
            del module
            captured_skip_prefixes.extend(skip_prefixes)

        def load_weights(self, weights: object, *, mapper: object) -> set[str]:
            del weights, mapper
            return set()

    pipeline_group = SimpleNamespace(is_first_rank=True)
    with (
        patch(
            "vllm.model_executor.models.apertus_mm.get_pp_group",
            return_value=pipeline_group,
        ),
        patch(
            "vllm.model_executor.models.apertus_mm.AutoWeightsLoader",
            CapturingWeightsLoader,
        ),
    ):
        model.load_weights(iter(()))

    assert "vision_tower." in captured_skip_prefixes
    assert "audio_tower." in captured_skip_prefixes


def test_extract_vision_code_ids_from_emu35_encode_output() -> None:
    expected_code_ids = torch.tensor([3, 5, 8])
    encode_output = (
        torch.zeros(1),
        0.0,
        (None, None, expected_code_ids),
    )

    actual_code_ids = Apertus1p5ForConditionalGeneration._extract_vision_code_ids(
        encode_output
    )

    assert torch.equal(actual_code_ids, expected_code_ids)


def test_extract_vision_code_ids_rejects_unexpected_output() -> None:
    with pytest.raises(ValueError, match="Emu3.5 encode output"):
        Apertus1p5ForConditionalGeneration._extract_vision_code_ids(
            (torch.zeros(1),)
        )
