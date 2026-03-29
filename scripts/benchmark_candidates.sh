#!/bin/bash
# ============================================================================
# Zero-shot benchmark of candidate base models for Japanese telephony STT
#
# Run on a RunPod A40 GPU. This script:
# 1. Installs jaeval + all model dependencies
# 2. Benchmarks 8 models on the v2 telephony corpus
# 3. Compares results
#
# Usage:
#   # On RunPod:
#   git clone https://github.com/egao0125/japanese-eval-V1.git
#   cd japanese-eval-V1
#   bash scripts/benchmark_candidates.sh
# ============================================================================

set -euo pipefail

echo "=== Japanese STT Model Selection Benchmark ==="
echo "Started: $(date -u)"
echo ""

# --- 1. Install dependencies ---
echo ">>> Installing jaeval + GPU dependencies..."
pip install -e ".[dev,gpu,pipeline,lenient]" 2>&1 | tail -3

# faster-whisper for local Whisper inference
pip install faster-whisper>=1.1.0 2>&1 | tail -1

# Check GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f}GB')"

TASK="tasks/stt/model_selection.yaml"
RESULTS_DIR="results/model_selection"
mkdir -p "$RESULTS_DIR"

# --- 2. Benchmark each model ---

echo ""
echo ">>> Benchmarking Whisper Large-v3-Turbo (809M)..."
jaeval benchmark "$TASK" --model whisper-turbo \
  --output "$RESULTS_DIR/whisper_turbo.json" 2>&1 | tail -5

echo ""
echo ">>> Benchmarking Whisper Large-v3 (1.55B)..."
jaeval benchmark "$TASK" --model whisper-v3 \
  --output "$RESULTS_DIR/whisper_v3.json" 2>&1 | tail -5

echo ""
echo ">>> Benchmarking Kotoba-Whisper v2.0 (756M)..."
jaeval benchmark "$TASK" --model kotoba-whisper \
  --output "$RESULTS_DIR/kotoba_whisper.json" 2>&1 | tail -5

echo ""
echo ">>> Benchmarking Qwen3-ASR-0.6B..."
# Qwen3-ASR uses HuggingFace transformers, not faster-whisper
jaeval benchmark "$TASK" --model qwen3-asr \
  --provider-arg model_id=Qwen/Qwen3-ASR-0.6B \
  --output "$RESULTS_DIR/qwen3_asr_06b.json" 2>&1 | tail -5

echo ""
echo ">>> Benchmarking Qwen3-ASR-1.7B..."
jaeval benchmark "$TASK" --model qwen3-asr \
  --provider-arg model_id=Qwen/Qwen3-ASR-1.7B \
  --output "$RESULTS_DIR/qwen3_asr_17b.json" 2>&1 | tail -5

echo ""
echo ">>> Benchmarking IBM Granite 4.0 1B Speech (Mar 2026)..."
# Granite uses HF transformers — needs custom provider or HF pipeline
pip install granite-speech 2>&1 | tail -1 || true
jaeval benchmark "$TASK" --model hf-inference \
  --provider-arg model_id=ibm-granite/granite-4.0-1b-speech \
  --output "$RESULTS_DIR/granite_speech.json" 2>&1 | tail -5

echo ""
echo ">>> Benchmarking Cohere Transcribe (Mar 2026, 2B)..."
jaeval benchmark "$TASK" --model hf-inference \
  --provider-arg model_id=CohereLabs/cohere-transcribe-03-2026 \
  --output "$RESULTS_DIR/cohere_transcribe.json" 2>&1 | tail -5

echo ""
echo ">>> Benchmarking NVIDIA Parakeet-TDT/CTC-0.6B-ja..."
pip install nemo_toolkit[asr] 2>&1 | tail -1 || true
jaeval benchmark "$TASK" --model hf-inference \
  --provider-arg model_id=nvidia/parakeet-tdt_ctc-0.6b-ja \
  --output "$RESULTS_DIR/parakeet_ja.json" 2>&1 | tail -5

# --- 3. Compare results ---

echo ""
echo "=== COMPARISON ==="
jaeval compare \
  "$RESULTS_DIR/whisper_turbo.json" \
  "$RESULTS_DIR/whisper_v3.json" \
  "$RESULTS_DIR/kotoba_whisper.json" \
  "$RESULTS_DIR/qwen3_asr_06b.json" \
  "$RESULTS_DIR/qwen3_asr_17b.json" \
  "$RESULTS_DIR/granite_speech.json" \
  "$RESULTS_DIR/cohere_transcribe.json" \
  "$RESULTS_DIR/parakeet_ja.json" \
  --output "$RESULTS_DIR/comparison.md"

cat "$RESULTS_DIR/comparison.md"

echo ""
echo "=== DONE ==="
echo "Finished: $(date -u)"
echo "Results in: $RESULTS_DIR/"
