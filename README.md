# japanese-eval-V1

Auto-research + evaluation harness for Japanese voice AI.

## What it does

Two modules in one repo:

1. **Evaluation Harness** -- YAML-driven benchmarking for Japanese STT/TTS/conversation systems (inspired by EleutherAI lm-evaluation-harness)
2. **Auto-Research** -- Agent-driven pipeline that discovers papers, models, and benchmarks for Japanese voice/speech AI (STORM/GPT-Researcher pattern)

## Quick Start

```bash
pip install -e ".[dev]"

# List available tasks and providers
jaeval list-tasks
jaeval list-models

# Run STT benchmark
jaeval benchmark tasks/stt/corpus_v2_clean.yaml --model deepgram

# Run auto-research
jaeval research "Japanese STT evaluation state of the art"

# Run LLM-as-judge on a call
jaeval judge --scorecard results/scorecards/call.json
```

## Architecture

```
src/jaeval/
├── core/                  # Japanese NLP: normalize, CER, hallucination, audio
├── harness/               # Evaluation harness
│   ├── providers/         # STT providers (deepgram, openai, whisper, qwen3, websocket)
│   ├── evaluators/        # Multi-tier eval (pipeline, LLM judge, scorecard, TTS)
│   ├── runner.py          # Benchmark orchestrator
│   ├── task.py            # YAML task definitions
│   ├── gate.py            # Pass/Warn/Fail thresholds
│   └── report.py          # Markdown + JSON output
└── research/              # Auto-research pipeline
    ├── orchestrator.py    # Plan → Search → Read → Synthesize
    ├── planner.py         # LLM question generation
    ├── searcher.py        # Parallel multi-source search
    └── sources/           # arxiv, GitHub, HuggingFace adapters
```

## YAML Task Definitions

Benchmarks are declarative YAML files:

```yaml
task: stt_corpus_v2_clean
type: stt
corpus:
  path: corpora/stt/corpus_v2
  ground_truth: ground_truth.json
pipeline:
  codec: null       # or g711_mulaw
  vad: null         # or silero
metrics: [cer, hallucination_count, latency_p50, latency_p90, rtf]
gates:
  median_cer: { pass: 0.05, warn: 0.08 }
  hallucinations: { pass: 0, warn: 2 }
```

## STT Providers

| Provider | Type | GPU | Setup |
|----------|------|-----|-------|
| deepgram | API | No | `DEEPGRAM_API_KEY` |
| openai | API | No | `OPENAI_API_KEY` |
| whisper | Local | Yes | `pip install .[gpu]` |
| qwen3-asr | Local | Yes | `pip install .[gpu]` + LoRA adapter |
| websocket | Remote | No | STT server URL |

## Optional Dependencies

```bash
pip install -e ".[research]"   # Auto-research (anthropic, arxiv)
pip install -e ".[gpu]"        # GPU providers (torch, faster-whisper)
pip install -e ".[lenient]"    # Lenient CER (fugashi, MeCab)
pip install -e ".[dev]"        # Dev tools (pytest, ruff, mypy)
```

## Tests

```bash
pytest tests/ -v
```

## Key Metrics

- **CER** (Character Error Rate) -- standard for Japanese (no word boundaries)
- **Lenient CER** -- accounts for kanji/hiragana/katakana spelling equivalence
- **Hallucination Count** -- kanji in hypothesis not in reference
- **Latency P50/P90** -- response time percentiles
- **RTF** -- Real-Time Factor (latency / audio duration)

## License

MIT
