## Project: japanese-eval-V1

Auto-research + evaluation harness for Japanese voice AI.

### Quick Start
```bash
pip install -e ".[dev]"
jaeval --help
```

### Run Tests
```bash
python -m pytest tests/   # 178 tests
ruff check src/ tests/
```

### CLI Commands
- `jaeval benchmark` — Run STT benchmark (YAML task + provider)
- `jaeval compare` — Compare benchmark results across providers/tasks
- `jaeval judge` — Run LLM-as-judge on a call transcript/scorecard
- `jaeval judge-compare` — Compare judge results across calls
- `jaeval eval` — Full call evaluation (Tier 1 scorecard + Tier 2 LLM judge)
- `jaeval research` — Auto-research pipeline
- `jaeval list-tasks` / `jaeval list-models` — Discovery commands

### Library API
```python
from jaeval import evaluate_call
result = evaluate_call("call_id", turns, run_judge=True)
```

### Architecture
- src/jaeval/core/ — Japanese NLP: normalize, CER, hallucination, audio
- src/jaeval/harness/ — Evaluation harness: YAML tasks, providers, runner
- src/jaeval/harness/evaluators/ — Multi-tier: scorecard, llm_judge, pipeline_eval
- src/jaeval/harness/providers/ — STT backends: deepgram, openai, whisper, qwen3, websocket
- src/jaeval/research/ — Auto-research: Plan → Search → Read → Synthesize
- src/jaeval/integration.py — High-level evaluate_call() for voice-fullduplex
- tasks/ — YAML benchmark task definitions
- corpora/ — Corpus data (audio gitignored, metadata tracked)
- examples/ — Integration examples
