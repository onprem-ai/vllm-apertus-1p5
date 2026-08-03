# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Apertus tokenizer helper wrappers for on-disk model weight loading.

Only the methods needed by apertus_mm.py are included here. The CPU-side
multimodal preprocessing (prompt layout, image resizing, audio normalization)
is handled by Apertus1p5MultiModalProcessor in apertus_mm.py, which uses
the HuggingFace processor directly -- not these helpers.

When transformers PR #47662 merges and Apertus1p5VisionTokenizerModel /
AutoModel for WavTokenizer become available, these helpers can be removed
entirely in favor of _init_component_model() (see apertus_mm.py docstring).
"""

from pathlib import Path
from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


class ApertusImageTokenizer:
    """Loads the Emu3.5 vision tokenizer from an on-disk checkpoint.

    Delegates to apertus_emu35.build_vision_tokenizer() which handles
    safetensors file discovery, weight prefix stripping, and model
    instantiation. Results are cached by (model_path, device, dtype).
    """

    _vision_tokenizer_cache: dict[tuple[str, str, torch.dtype], Any] = {}

    def __init__(self, vision_config: dict[str, Any] | None = None) -> None:
        self.vision_config = vision_config or {}

    def load_vision_tokenizer(
        self,
        model_path: str,
        device: str,
        dtype: torch.dtype,
        vision_config: dict[str, Any],
    ) -> Any:
        from vllm.model_executor.models.apertus_emu35 import build_vision_tokenizer

        path = Path(model_path).expanduser()
        assert path.exists() and path.is_dir(), (
            f"Model directory {model_path} does not exist or is not a directory."
        )
        control_file = path / "model-vision_tokenizer-model.safetensors"
        assert control_file.is_file(), (
            "Vision tokenizer file model-vision_tokenizer-model.safetensors "
            f"must exist in {model_path}."
        )

        if device == "cpu" and dtype in (torch.float16, torch.bfloat16):
            dtype = torch.float32

        cache_key = (model_path, device, dtype)
        if cache_key in self._vision_tokenizer_cache:
            return self._vision_tokenizer_cache[cache_key]

        logger.info(
            "[Apertus MM] loading Emu3.5 vision tokenizer on device=%r", device
        )
        vision_tokenizer = build_vision_tokenizer(
            type="ibq",
            model_path=str(path.resolve()),
            device=device,
            vision_config=vision_config,
        )
        if isinstance(dtype, torch.dtype):
            vision_tokenizer = vision_tokenizer.to(dtype=dtype)

        self._vision_tokenizer_cache[cache_key] = vision_tokenizer
        return vision_tokenizer


class ApertusAudioTokenizer:
    """Loads the WavTokenizer audio encoder from an on-disk checkpoint.

    Delegates to apertus_wavetokenizer.build_audio_tokenizer() which handles
    safetensors file discovery, model instantiation, and weight loading.
    Results are cached by (model_path, device, compile_flag).
    """

    _audio_tokenizer_cache: dict[tuple[str, str, bool], Any] = {}

    def __init__(self, audio_config: dict[str, Any] | None = None) -> None:
        self.audio_config = audio_config or {}

    @staticmethod
    def _coerce_bool(value: object, *, default: bool) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "t", "yes", "y", "on"}:
                return True
            if lowered in {"0", "false", "f", "no", "n", "off"}:
                return False
        return default

    def load_audio_tokenizer(
        self,
        model_path: str,
        device: str,
        audio_config: dict[str, Any],
    ) -> Any:
        from vllm.model_executor.models.apertus_wavetokenizer import (
            build_audio_tokenizer,
        )

        path = Path(model_path).expanduser()
        assert path.exists() and path.is_dir(), (
            f"Model directory {model_path} does not exist or is not a directory."
        )
        control_file = path / "model-wavtokenizer-model.safetensors"
        assert control_file.is_file(), (
            "Audio tokenizer file model-wavtokenizer-model.safetensors must "
            f"exist in {model_path}."
        )

        tokenizer_compile = self._coerce_bool(
            audio_config.get("apertus_audio_tokenizer_compile"), default=False
        )
        cache_key = (model_path, device, tokenizer_compile)
        if cache_key in self._audio_tokenizer_cache:
            return self._audio_tokenizer_cache[cache_key]

        logger.info(
            "[Apertus MM] loading WavTokenizer audio tokenizer on device=%r", device
        )
        audio_tokenizer = build_audio_tokenizer(
            type="wavtokenizer",
            model_path=str(path.resolve()),
            device=device,
            audio_config=audio_config,
        )
        self._audio_tokenizer_cache[cache_key] = audio_tokenizer
        return audio_tokenizer
