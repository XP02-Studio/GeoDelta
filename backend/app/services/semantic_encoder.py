from __future__ import annotations

import importlib
import os
from typing import Callable, Sequence

VECTOR_SIZE = 512


def mock_rsclip_encoder(text: str) -> list[float]:
    """Fallback dummy encoder generating a 512-D vector for local testing."""
    return [0.042] * VECTOR_SIZE


class RSCLIPTextEncoder:
    """Thin adapter around the locally installed RS-CLIP text encoder.
    Set ``RSCLIP_ENCODER`` to ``module:function``.
    """

    def __init__(self, encoder: Callable[[str], Sequence[float]] | None = None) -> None:
        self._encoder = encoder or self._load_configured_encoder()

    @staticmethod
    def _load_configured_encoder() -> Callable[[str], Sequence[float]]:
        target = os.getenv("RSCLIP_ENCODER")
        if not target or ":" not in target:
            # Fallback to mock encoder if env var is empty
            return mock_rsclip_encoder

        module_name, function_name = target.split(":", 1)
        try:
            mod = importlib.import_module(module_name)
            function = getattr(mod, function_name, None)
            if not callable(function):
                raise RuntimeError(f"Configured RS-CLIP encoder is not callable: {target}")
            return function
        except (ImportError, AttributeError):
            return mock_rsclip_encoder

    def encode(self, text: str) -> list[float]:
        vector = list(self._encoder(text))
        if len(vector) != VECTOR_SIZE:
            raise ValueError(f"RS-CLIP output must contain {VECTOR_SIZE} values")
        return [float(value) for value in vector]