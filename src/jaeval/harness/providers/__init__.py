"""STT provider registry.

Provides ``get_provider()`` to look up providers by name and
``PROVIDER_REGISTRY`` mapping names to classes.
"""

from __future__ import annotations

from .base import STTProvider, TranscribeResult
from .deepgram import DeepgramProvider
from .openai_stt import OpenAISTTProvider
from .whisper import WhisperProvider
from .qwen3 import Qwen3ASRProvider
from .websocket_stt import WebSocketSTTProvider

PROVIDER_REGISTRY: dict[str, type[STTProvider]] = {
    "deepgram": DeepgramProvider,
    "openai": OpenAISTTProvider,
    "whisper": WhisperProvider,
    "qwen3-asr": Qwen3ASRProvider,
    "websocket": WebSocketSTTProvider,
}


def get_provider(name: str, **kwargs) -> STTProvider:
    """Instantiate an STT provider by name.

    Args:
        name: Provider name (must be a key in PROVIDER_REGISTRY).
        **kwargs: Passed to the provider constructor.

    Returns:
        An initialized STTProvider instance.

    Raises:
        KeyError: If the provider name is not registered.
    """
    if name not in PROVIDER_REGISTRY:
        available = ", ".join(sorted(PROVIDER_REGISTRY.keys()))
        raise KeyError(f"Unknown provider '{name}'. Available: {available}")

    cls = PROVIDER_REGISTRY[name]
    return cls(**kwargs)


__all__ = [
    "STTProvider",
    "TranscribeResult",
    "DeepgramProvider",
    "OpenAISTTProvider",
    "WhisperProvider",
    "Qwen3ASRProvider",
    "WebSocketSTTProvider",
    "PROVIDER_REGISTRY",
    "get_provider",
]
