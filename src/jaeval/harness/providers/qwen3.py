"""Qwen3-ASR STT provider (base model + optional LoRA adapter).

Supports both the base Qwen3-ASR-1.7B model and fine-tuned variants
loaded via PEFT LoRA adapter merging.

Extracted from voice-fullduplex ``scripts/eval/benchmark_stt.py``
(``stt_qwen3_asr`` and ``stt_qwen3_asr_lora`` functions).
"""

from __future__ import annotations

import time

from .base import STTProvider, TranscribeResult


class Qwen3ASRProvider(STTProvider):
    """Qwen3-ASR-1.7B provider with optional LoRA adapter.

    When ``adapter_path`` is ``None``, uses the high-level
    ``qwen_asr.Qwen3ASRModel`` API for simple inference.

    When ``adapter_path`` is set, loads the base model via the
    transformers backend, applies the LoRA adapter with PEFT,
    merges weights, and runs manual generate + decode.

    Args:
        adapter_path: Path to a LoRA adapter directory. If ``None``,
            uses the base model without fine-tuning.
        base_model_id: HuggingFace model ID for the base model.
        max_new_tokens: Maximum tokens to generate per utterance.
    """

    name = "qwen3-asr"
    requires_gpu = True

    def __init__(
        self,
        *,
        adapter_path: str | None = None,
        base_model_id: str = "Qwen/Qwen3-ASR-1.7B",
        max_new_tokens: int = 256,
    ):
        self.adapter_path = adapter_path
        self.base_model_id = base_model_id
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None
        self._mode: str = "base"  # "base" or "lora"

    def setup(self) -> None:
        """Load the model (base or with LoRA adapter merged)."""
        import torch

        if self.adapter_path is not None:
            self._setup_lora(torch)
        else:
            self._setup_base(torch)

    def _setup_base(self, torch) -> None:
        """Load the base Qwen3-ASR model using the high-level API."""
        from qwen_asr import Qwen3ASRModel

        self._model = Qwen3ASRModel.from_pretrained(
            self.base_model_id,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            max_new_tokens=self.max_new_tokens,
        )
        self._mode = "base"

    def _setup_lora(self, torch) -> None:
        """Load the base model, apply LoRA adapter, and merge weights."""
        from qwen_asr.core.transformers_backend.processing_qwen3_asr import (
            Qwen3ASRProcessor,
        )
        from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
            Qwen3ASRForConditionalGeneration,
        )
        from peft import PeftModel

        processor = Qwen3ASRProcessor.from_pretrained(self.base_model_id)
        outer_model = Qwen3ASRForConditionalGeneration.from_pretrained(
            self.base_model_id,
            dtype=torch.float16,
        )

        # Apply LoRA to the thinker sub-model (which has forward/generate)
        thinker_with_lora = PeftModel.from_pretrained(
            outer_model.thinker,
            self.adapter_path,
        )
        thinker_with_lora = thinker_with_lora.merge_and_unload()
        outer_model.thinker = thinker_with_lora
        outer_model.thinker.eval()

        self._model = outer_model
        self._processor = processor
        self._mode = "lora"

    def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000) -> TranscribeResult:
        """Transcribe audio using Qwen3-ASR.

        Dispatches to the appropriate inference path based on whether
        a LoRA adapter was loaded.

        Args:
            audio_bytes: WAV file bytes (including header).
            sample_rate: Sample rate of the audio in Hz.

        Returns:
            TranscribeResult with transcript and inference latency.
        """
        if self._model is None:
            self.setup()

        if self._mode == "lora":
            return self._transcribe_lora(audio_bytes, sample_rate)
        return self._transcribe_base(audio_bytes, sample_rate)

    def _transcribe_base(self, audio_bytes: bytes, sample_rate: int) -> TranscribeResult:
        """Transcribe using the high-level Qwen3ASRModel API."""
        from jaeval.core.audio import decode_wav_to_float32

        audio = decode_wav_to_float32(audio_bytes, target_sr=16000)

        t0 = time.monotonic()
        results = self._model.transcribe(
            audio=(audio, 16000),
            language="Japanese",
        )
        transcript = results[0].text if results else ""
        latency = time.monotonic() - t0

        return TranscribeResult(
            text=transcript,
            latency_sec=latency,
            metadata={
                "model_id": self.base_model_id,
                "mode": "base",
            },
        )

    def _transcribe_lora(self, audio_bytes: bytes, sample_rate: int) -> TranscribeResult:
        """Transcribe using the merged LoRA model with manual generation."""
        import torch
        from jaeval.core.audio import decode_wav_to_float32

        audio = decode_wav_to_float32(audio_bytes, target_sr=16000)

        device = next(self._model.thinker.parameters()).device
        dtype = next(self._model.thinker.parameters()).dtype

        # Build prompt (same format as training -- forced Japanese language)
        prompt_text = (
            "<|im_start|>system\n<|im_end|>\n"
            "<|im_start|>user\n"
            "<|audio_start|><|audio_pad|><|audio_end|>"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            "language Japanese<asr_text>"
        )

        inputs = self._processor(
            text=[prompt_text],
            audio=[audio],
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        if "input_features" in inputs:
            inputs["input_features"] = inputs["input_features"].to(dtype)

        prompt_len = inputs["input_ids"].shape[1]

        t0 = time.monotonic()
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs, max_new_tokens=self.max_new_tokens
            )
        latency = time.monotonic() - t0

        if hasattr(outputs, "sequences"):
            gen_ids = outputs.sequences[:, prompt_len:]
        else:
            gen_ids = outputs[:, prompt_len:]

        transcript = self._processor.tokenizer.batch_decode(
            gen_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return TranscribeResult(
            text=transcript,
            latency_sec=latency,
            metadata={
                "model_id": self.base_model_id,
                "adapter_path": self.adapter_path,
                "mode": "lora",
            },
        )

    def teardown(self) -> None:
        """Release GPU memory."""
        self._model = None
        self._processor = None
