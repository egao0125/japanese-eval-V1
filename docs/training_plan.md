# Japanese Telephony STT Fine-Tuning Plan

## Objective

Fine-tune a fresh base model for Japanese telephony STT (8kHz, noisy, conversational keigo), targeting < 5% median CER on corpus v2 and < 3% on clean speech.

## Phase 1: Model Selection (Zero-Shot Benchmark)

Run `scripts/benchmark_candidates.sh` on RunPod A40. Candidates:

| Model | Params | HF ID | Why |
|-------|--------|-------|-----|
| Whisper Large-v3-Turbo | 809M | `deepdml/faster-whisper-large-v3-turbo-ct2` | Best LoRA ecosystem, 4 dec layers, fast |
| Whisper Large-v3 | 1.55B | `large-v3` | Gold standard baseline |
| Kotoba-Whisper v2.0 | 756M | `kotoba-tech/kotoba-whisper-v2.0-faster` | Pre-trained on 7.2M Japanese clips |
| Qwen3-ASR-0.6B | 600M | `Qwen/Qwen3-ASR-0.6B` | Newest SOTA, streaming, Apache-2.0 |
| Qwen3-ASR-1.7B | 1.7B | `Qwen/Qwen3-ASR-1.7B` | Best Fleurs CER (5.2%) |
| IBM Granite 4.0 1B Speech | 1B | `ibm-granite/granite-4.0-1b-speech` | 16,700h JP data, keyword biasing, fine-tune notebook, #1 OpenASR |
| Cohere Transcribe | 2B | `CohereLabs/cohere-transcribe-03-2026` | #1 HF ASR leaderboard (WER 5.42), 0.5M hours training, Fast-Conformer |
| NVIDIA Parakeet-TDT/CTC-0.6B-ja | 600M | `nvidia/parakeet-tdt_ctc-0.6b-ja` | Best raw Japanese CER (6.4% JSUT), NeMo native |

**Eliminated candidates:**
- Microsoft VibeVoice ASR (9B params) — too large for A40 LoRA fine-tuning

**Selection criteria (in order):**
1. Lowest median CER on telephony corpus v2 (zero-shot)
2. Lowest hallucination count
3. Fastest RTF (real-time factor)
4. Fine-tuning friendliness (LoRA support, training framework)

## Phase 2: Data Preparation

### Training Data Sources

1. **ReazonSpeech** — 35,000+ hours Japanese speech (primary pre-training corpus)
   - Download via `reazon-research/ReazonSpeech` HuggingFace dataset
   - Filter for conversational segments

2. **Real telephony data** — 465 customer segments from voice-fullduplex
   - Already processed: VAD → MFCC dedup → transcribe → speaker classify
   - Path: `/Users/ozawaegao/voice-fullduplex/data/`
   - v6 dataset: 17,615 train / 1,970 eval

3. **Synthetic telephony augmentation** — Apply to clean speech data:
   - G.711 mu-law codec simulation (jaeval's `pipeline_eval.py`)
   - Background noise mixing (office, street, crowd)
   - 8kHz→16kHz resampling artifacts

### Augmentation Pipeline

```python
from jaeval.harness.evaluators.pipeline_eval import PipelineEvaluator, PipelineConfig

config = PipelineConfig(codec="g711_mulaw", energy_gate_rms=0.01)
pipeline = PipelineEvaluator(config)

# Apply telephony degradation to clean training data
processed_audio, stats = pipeline.process_audio(clean_audio_bytes, sample_rate=16000)
```

### Data Split

| Split | Source | Hours (target) |
|-------|--------|---------------|
| Train | ReazonSpeech (filtered) + real telephony + augmented | ~500h |
| Eval | Corpus v2 (100 utts) + corpus v3 | ~2h |
| Test | Held-out real calls (unseen) | ~1h |

## Phase 3: Fine-Tuning

### LoRA Configuration (starting point)

```yaml
# Adjust based on Phase 1 winner
lora:
  r: 64                    # Rank (v3 rsLoRA broke plateau at 64)
  alpha: 128               # Scaling (2x rank)
  dropout: 0.05
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
  use_rslora: true         # Rank-stabilized LoRA
  use_dora: false          # Start without DoRA, add if plateau

training:
  epochs: 3
  batch_size: 8            # A40 46GB VRAM
  gradient_accumulation: 4  # Effective batch size 32
  learning_rate: 2e-4
  lr_scheduler: cosine
  warmup_ratio: 0.05
  fp16: true
  max_audio_length_sec: 30
  save_steps: 500
  eval_steps: 500
```

### Training Framework

For Whisper-based models:
```bash
# HuggingFace Transformers + PEFT
pip install transformers peft datasets accelerate
python scripts/train/train_whisper_lora.py \
  --model_id openai/whisper-large-v3-turbo \
  --dataset_path /workspace/data/train \
  --output_dir /workspace/checkpoints/whisper-turbo-ja-v1 \
  --lora_r 64 --lora_alpha 128 \
  --epochs 3 --batch_size 8
```

For Qwen3-ASR:
```bash
# Uses Qwen's own training recipe
pip install qwen-asr peft
# Follow: https://huggingface.co/Qwen/Qwen3-ASR-0.6B#fine-tuning
```

For IBM Granite 4.0 1B Speech:
```bash
# Official fine-tune notebook: https://github.com/ibm-granite/granite-speech
pip install granite-speech transformers peft datasets accelerate
# Conformer encoder + Granite-4.0-1b decoder — LoRA targets decoder layers
# Keyword biasing: supply domain terms ("レコ", "StepAI") via prompt context
python scripts/train/train_granite_lora.py \
  --model_id ibm-granite/granite-4.0-1b-speech \
  --dataset_path /workspace/data/train \
  --output_dir /workspace/checkpoints/granite-ja-v1 \
  --lora_r 64 --lora_alpha 128 \
  --epochs 3 --batch_size 8
```

For Cohere Transcribe:
```bash
# Fast-Conformer X-attention encoder-decoder, 2B params
# Apache-2.0, standard HF Transformers PEFT workflow
pip install transformers peft datasets accelerate
python scripts/train/train_cohere_lora.py \
  --model_id CohereLabs/cohere-transcribe-03-2026 \
  --dataset_path /workspace/data/train \
  --output_dir /workspace/checkpoints/cohere-ja-v1 \
  --lora_r 64 --lora_alpha 128 \
  --epochs 3 --batch_size 4  # 2B params — reduce batch for A40
```

## Phase 4: Evaluation Loop

After each checkpoint:

```bash
# Run jaeval benchmark
jaeval benchmark tasks/stt/model_selection.yaml \
  --model whisper \
  --provider-arg model_id=/workspace/checkpoints/latest \
  --output results/finetune/checkpoint_N.json

# Compare to baseline
jaeval compare results/model_selection/baseline.json results/finetune/checkpoint_N.json

# Category breakdown
jaeval benchmark tasks/stt/keigo.yaml --model whisper --provider-arg model_id=/workspace/checkpoints/latest
jaeval benchmark tasks/stt/short.yaml --model whisper --provider-arg model_id=/workspace/checkpoints/latest
jaeval benchmark tasks/stt/number.yaml --model whisper --provider-arg model_id=/workspace/checkpoints/latest
```

### Gates (must pass before deployment)

| Metric | Pass | Warn | Fail |
|--------|------|------|------|
| Median CER (overall) | < 5% | < 8% | >= 8% |
| Median CER (keigo) | < 5% | < 10% | >= 10% |
| Median CER (short) | < 10% | < 25% | >= 25% |
| Hallucinations | 0 | <= 2 | > 2 |
| Latency P90 | < 1.5s | < 2.0s | >= 2.0s |
| RTF | < 0.5 | < 1.0 | >= 1.0 |

## Phase 5: Iteration

If gates fail:
1. Identify failing category (keigo? numbers? short utterances?)
2. Research targeted augmentation for that category
3. Increase training data for weak category
4. Adjust LoRA rank or try DoRA
5. Retrain and re-benchmark

## RunPod Setup

```bash
# A40 (46GB VRAM) recommended
# Estimated time: ~4h for full training run

# SSH in, then:
git clone https://github.com/egao0125/japanese-eval-V1.git
cd japanese-eval-V1
pip install -e ".[dev,gpu,pipeline,lenient]"
pip install transformers peft datasets accelerate faster-whisper

# Copy training data
scp -r /path/to/training/data root@<runpod-ip>:/workspace/data/

# Run Phase 1 benchmark
bash scripts/benchmark_candidates.sh

# Select winner, then fine-tune (Phase 3)
```

## Key Lessons from v3-v9 (Qwen3-ASR fine-tuning)

- Rank 64 + MLP + rsLoRA broke the plateau (v3)
- DoRA α=128 got best clean CER but had keigo instability
- ~70% production clean rate ceiling was NOT a model problem — it was a pipeline problem
- Energy gate 0.01 eliminated hallucinations completely
- Pre-roll 2.0s was the biggest lever for CER improvement
- NEVER compare CER across different corpora
