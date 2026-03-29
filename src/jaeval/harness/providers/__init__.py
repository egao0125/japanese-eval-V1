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
from .hf_inference import HFInferenceProvider

PROVIDER_REGISTRY: dict[str, type[STTProvider]] = {
    "deepgram": DeepgramProvider,
    "openai": OpenAISTTProvider,
    "whisper": WhisperProvider,
    "qwen3-asr": Qwen3ASRProvider,
    "websocket": WebSocketSTTProvider,
    "hf-inference": HFInferenceProvider,
}

# Convenience aliases for specific models
MODEL_ALIASES: dict[str, tuple[str, dict]] = {
    "whisper-turbo": ("whisper", {"model_id": "deepdml/faster-whisper-large-v3-turbo-ct2"}),
    "whisper-v3": ("whisper", {"model_id": "large-v3"}),
    "kotoba-whisper": ("whisper", {"model_id": "kotoba-tech/kotoba-whisper-v2.0-faster"}),
    "kotoba-whisper-v2.2": ("whisper", {"model_id": "kotoba-tech/kotoba-whisper-v2.2-faster"}),
    "hf-whisper-turbo": ("hf-inference", {"model_id": "openai/whisper-large-v3-turbo"}),
    "hf-kotoba": ("hf-inference", {"model_id": "kotoba-tech/kotoba-whisper-v2.0"}),
    "hf-qwen3-asr": ("hf-inference", {"model_id": "Qwen/Qwen3-ASR-0.6B"}),
    "hf-granite-speech": ("hf-inference", {"model_id": "ibm-granite/granite-4.0-1b-speech"}),
    "hf-cohere-transcribe": ("hf-inference", {"model_id": "CohereLabs/cohere-transcribe-03-2026"}),
}


def get_provider(name: str, **kwargs) -> STTProvider:
    """Instantiate an STT provider by name or alias.

    Supports both direct provider names (e.g. ``whisper``) and
    convenience aliases (e.g. ``whisper-turbo``, ``hf-kotoba``).

    Args:
        name: Provider name or alias.
        **kwargs: Passed to the provider constructor (overrides alias defaults).

    Returns:
        An initialized STTProvider instance.

    Raises:
        KeyError: If the provider name is not registered.
    """
    # Resolve alias to (provider_name, default_kwargs)
    if name in MODEL_ALIASES:
        base_name, alias_kwargs = MODEL_ALIASES[name]
        merged = {**alias_kwargs, **kwargs}
        cls = PROVIDER_REGISTRY[base_name]
        provider = cls(**merged)
        provider.name = name  # Use alias as display name
        return provider

    if name not in PROVIDER_REGISTRY:
        available = sorted(set(list(PROVIDER_REGISTRY.keys()) + list(MODEL_ALIASES.keys())))
        raise KeyError(f"Unknown provider '{name}'. Available: {', '.join(available)}")

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
    "HFInferenceProvider",
    "PROVIDER_REGISTRY",
    "MODEL_ALIASES",
    "get_provider",
]
