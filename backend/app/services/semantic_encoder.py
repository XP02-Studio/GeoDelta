from __future__ import annotations

import importlib
import os
from typing import Callable, Sequence

VECTOR_SIZE = 512


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
            def unavailable_encoder(_: str) -> Sequence[float]:
                raise RuntimeError(
                    "RSCLIP_ENCODER must be set to module:function for semantic search. "
                    "Refusing to use a dummy embedding because it returns the same result for every query."
                )

            return unavailable_encoder

        module_name, function_name = target.split(":", 1)
        try:
            mod = importlib.import_module(module_name)
            function = getattr(mod, function_name, None)
            if not callable(function):
                raise RuntimeError(f"Configured RS-CLIP encoder is not callable: {target}")
            return function
        except (ImportError, AttributeError) as exc:
            def unavailable_encoder(_: str, error: Exception = exc) -> Sequence[float]:
                raise RuntimeError(f"Unable to load RSCLIP_ENCODER '{target}': {error}") from error

            return unavailable_encoder

    def encode(self, text: str) -> list[float]:
        vector = list(self._encoder(text))
        if len(vector) != VECTOR_SIZE:
            raise ValueError(f"RS-CLIP output must contain {VECTOR_SIZE} values")
        return [float(value) for value in vector]
