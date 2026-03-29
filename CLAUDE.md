## Project: japanese-eval-V1

Auto-research + evaluation harness for Japanese voice AI.

### Quick Start
pip install -e ".[dev]"
jaeval --help

### Run Tests
pytest tests/

### Architecture
- src/jaeval/core/ — Japanese NLP: normalize, CER, hallucination, audio
- src/jaeval/harness/ — Evaluation harness: YAML tasks, providers, runner
- src/jaeval/research/ — Auto-research: agent pipeline for paper/model discovery
- tasks/ — YAML benchmark task definitions
- corpora/ — Corpus data (audio gitignored, metadata tracked)
