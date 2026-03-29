# Zero-Shot Model Selection Benchmark

**Corpus**: v2 — 100 real telephony utterances (8kHz, noisy, conversational keigo)
**Date**: 2026-03-29
**GPU**: NVIDIA A40 (46GB VRAM, RunPod)

## Overall Rankings

| Rank | Model | Params | Median CER | Mean CER | Halluc. | Lat P50 | RTF |
|------|-------|--------|-----------|---------|---------|---------|-----|
| 1 | **Qwen3-ASR-1.7B** | 1.7B | **16.7%** | 36.2% | 53 | 0.41s | 0.198 |
| 2 | Whisper-v3 | 1.55B | 20.0% | 37.1% | 43 | 0.29s | 0.172 |
| 3 | Whisper-Turbo | 809M | 22.5% | 42.4% | 40 | **0.18s** | **0.110** |
| 4 | Qwen3-ASR-0.6B | 600M | 22.6% | 38.9% | 51 | 0.39s | 0.187 |
| 5 | Granite 4.0 1B Speech | 1B | 27.3% | 42.8% | **25** | 0.63s | N/A |
| 6 | Kotoba-Whisper v2.0 | 756M | 28.7% | 40.5% | 47 | 0.15s | 0.095 |

**API baselines** (not fine-tunable, for reference):
| | Deepgram Nova | API | 15.0% | **24.2%** | **29** | 0.65s | 0.540 |
| | OpenAI Whisper | API | **14.8%** | 37.6% | 48 | 0.85s | 0.592 |

## Per-Category Breakdown (mean CER)

| Model | greeting | keigo | number | proper_noun | compound | short |
|-------|---------|-------|--------|------------|----------|-------|
| **Qwen3-ASR-1.7B** | **8.8%** | 29.5% | 20.6% | 31.8% | 28.3% | 81.6% |
| Whisper-v3 | 28.5% | **27.4%** | 20.7% | 32.0% | **26.1%** | 75.2% |
| Whisper-Turbo | 24.1% | 28.5% | 19.7% | 34.9% | 27.9% | 100.0% |
| Qwen3-ASR-0.6B | 21.5% | 33.1% | 23.2% | 33.7% | 28.9% | 78.4% |
| Granite 4.0 1B | 27.8% | 41.2% | 31.1% | 33.0% | 29.8% | 78.9% |
| Kotoba-Whisper v2.0 | 39.9% | 36.6% | **16.7%** | 44.6% | 27.9% | **67.0%** |

## Analysis

### Winner: Qwen3-ASR-1.7B
- **Lowest median CER** (16.7%) among all local models
- **Best greeting CER** (8.8%) — important for telephony
- Strong across all non-short categories
- Already has proven LoRA fine-tuning ecosystem (v3-v9 in voice-fullduplex)

### Notable findings:
1. **Short utterances are catastrophic for ALL models** (67-100% CER) — likely a corpus/normalization issue
2. **Granite has fewest hallucinations** (25) despite higher CER — suggests conservative output
3. **Kotoba-Whisper underperforms** despite 7.2M Japanese clips — may need telephony-specific fine-tuning
4. **Whisper-Turbo is fastest** (RTF 0.110) but sacrifices accuracy
5. **All models FAIL the 5% median CER gate** — fine-tuning is essential

### Recommendation
Fine-tune **Qwen3-ASR-1.7B** as primary candidate:
- Best zero-shot CER on telephony data
- Proven fine-tuning path (rsLoRA rank 64 worked in v3-v9)
- 1.7B params fits comfortably on A40 for LoRA training

Secondary candidate: **Whisper-v3** (1.55B)
- Strong LoRA ecosystem (HuggingFace PEFT)
- Competitive CER, fewer hallucinations than Qwen3

### Skipped models
- **Cohere Transcribe**: Gated model, requires HF approval
- **NVIDIA Parakeet-0.6B-ja**: Requires NeMo toolkit (too large for 20GB disk)
- **Microsoft VibeVoice**: 9B params, too large for A40 LoRA
