from __future__ import annotations

import importlib
import os
from typing import Any, Callable, Sequence


VECTOR_SIZE = 512


class RSCLIPTextEncoder:
    """Thin adapter around the locally installed RS-CLIP text encoder.

    Set ``RSCLIP_ENCODER`` to ``module:function``. The function receives a
    plain-English query and must return 512 numeric values.
    """

    def __init__(self, encoder: Callable[[str], Sequence[float]] | None = None) -> None:
        self._encoder = encoder or self._load_configured_encoder()

    @staticmethod
    def _load_configured_encoder() -> Callable[[str], Sequence[float]]:
        target = os.getenv("RSCLIP_ENCODER")
        if not target or ":" not in target:
            raise RuntimeError(
                "RSCLIP_ENCODER must point to a local module:function encoder"
            )
        module_name, function_name = target.split(":", 1)
        function = getattr(importlib.import_module(module_name), function_name, None)
        if not callable(function):
            raise RuntimeError(f"Configured RS-CLIP encoder is not callable: {target}")
        return function

    def encode(self, text: str) -> list[float]:
        vector = list(self._encoder(text))
        if len(vector) != VECTOR_SIZE:
            raise ValueError(f"RS-CLIP output must contain {VECTOR_SIZE} values")
        return [float(value) for value in vector]