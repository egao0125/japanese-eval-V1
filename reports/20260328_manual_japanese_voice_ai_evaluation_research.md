# Japanese Voice AI Evaluation: State of the Art 2024-2026

## Executive Summary

This report surveys the current landscape of Japanese voice AI evaluation — covering ASR benchmarks, evaluation metrics, LLM-as-judge methods, and auto-research frameworks. We find a maturing ecosystem around Japanese ASR evaluation (driven by ReazonSpeech and kotoba-whisper), growing adoption of CER over WER for Japanese, and a rapidly evolving LLM-as-judge paradigm for conversation quality assessment. Voice agent evaluation is emerging as a distinct discipline with dedicated platforms (Hamming AI, Braintrust) and benchmarks (VoiceAgentBench).

---

## 1. Japanese ASR Evaluation Metrics

### Character Error Rate (CER) as the Standard

CER is now the consensus metric for Japanese ASR evaluation. Unlike WER (Word Error Rate), CER avoids the word segmentation problem inherent in Japanese text. A 2025 NAACL Findings paper — *"Advocating Character Error Rate for Multilingual ASR Evaluation"* — formally argues that CER should be prioritized in multilingual ASR evaluations to account for varying linguistic characteristics.

**Key insight**: Standard CER penalizes valid spelling variations (e.g., 漢字 vs かんじ vs カンジ). The ACL 2023 paper *"Lenient Evaluation of Japanese Speech Recognition: Modeling Naturally Occurring Spelling Inconsistency"* (arxiv:2306.04530) addresses this with a lenient CER method that does not penalize valid alternate spellings, achieving 2.4-3.1% absolute CER reduction. Human raters validated 95.4% of proposed spelling variants as plausible.

### Other Metrics
- **Hallucination Count**: Kanji characters in ASR output not present in reference — critical for business telephony where fabricated characters cause misunderstanding
- **Latency P50/P90**: Response time percentiles, with telephony targets of 550ms (P50) and 1700ms (P90)
- **RTF (Real-Time Factor)**: Latency / audio duration — values <1.0 indicate real-time capability

---

## 2. Japanese ASR Benchmarks and Corpora

### ReazonSpeech
The largest free Japanese ASR corpus, created by Reazon Holdings (research.reazon.jp). Built from TV broadcast audio with automatic transcription quality filtering (CER ≤ 0.33 threshold). Provides tools for evaluating ASR models and creating Japanese audio corpora. The corpus is used as the standard training and evaluation dataset for Japanese ASR research.

- **GitHub**: github.com/reazon-research/ReazonSpeech
- **Scale**: Massive (millions of utterances from TV broadcasts)
- **Use**: Training + held-out test splits for benchmarking

### Kotoba-Whisper (kotoba-tech)
Distilled Japanese ASR models built on OpenAI Whisper large-v3, trained on ReazonSpeech:
- **kotoba-whisper-v1.0**: 6.3x faster than whisper-large-v3 with comparable CER
- **kotoba-whisper-v2.0**: Improved v2 with competitive CER/WER on out-of-domain sets (JSUT, CommonVoice 8.0 Japanese)
- **Benchmark suites**: HuggingFace org `japanese-asr` hosts all models, datasets, and evaluation scripts
- **Limitation**: CER can reach 12.2% and WER 56.4% on out-of-domain sets

### Multi-Pass Augmented GER Benchmark
The 2024 paper *"Benchmarking Japanese Speech Recognition on ASR-LLM Setups with Multi-Pass Augmented Generative Error Correction"* (arxiv:2408.16180) presents the first GER (Generative Error Correction) benchmark for Japanese ASR. It evaluates how LLM-based error correction can enhance Japanese ASR using the CSJ (Corpus of Spontaneous Japanese) evaluation sets.

### Efficient Adaptation of Multilingual Models (arxiv:2412.10705)
A 2024 paper exploring LoRA fine-tuning of multilingual Whisper models for Japanese ASR, evaluating the improvement achieved after fine-tuning compared to baseline performance.

### Standard Test Sets
| Dataset | Type | Size | Usage |
|---------|------|------|-------|
| ReazonSpeech test | Broadcast | Large | In-domain benchmark |
| JSUT Basic 5000 | Read speech | 5000 utts | Out-of-domain test |
| CommonVoice 8.0 (ja) | Crowd-sourced | Medium | Cross-domain test |
| CSJ Eval {1,2,3} | Spontaneous | ~2.6k utts | Academic standard |

---

## 3. LLM-as-Judge for Conversation Evaluation

### The Paradigm
LLM-as-Judge evaluates text/conversation quality using an LLM as the evaluator. The judge reads the full conversation and scores it against defined metrics. This is now the dominant approach for evaluating voice agents where traditional metrics (WER/CER) are insufficient.

### Key Developments
1. **Multi-Turn Dialogue Evaluation** (arxiv:2508.00454): *"Learning an Efficient Multi-Turn Dialogue Evaluator from Multiple LLM Judges"* — trains efficient evaluators from multiple LLM judge perspectives.
2. **Agent-as-a-Judge** (arxiv:2508.02994): Extends LLM-as-judge to multi-agent systems that evaluate processes (not just outputs), including factual correctness, readability, and appropriateness.
3. **Communication Systems** (arxiv:2510.12462): *"Evaluating and Mitigating LLM-as-a-judge Bias in Communication Systems"* — addresses bias in LLM judges for subjective evaluation tasks.

### Production Frameworks
| Platform | Approach | Key Feature |
|----------|----------|-------------|
| Langfuse | Scoring rubrics | Open-source, trace-based evaluation |
| LangChain | Human-calibrated | Feedback loop for judge calibration |
| Confident AI | Multi-metric | Task completion, role adherence, conversation completeness |
| Braintrust | Dataset-driven | Cross-run comparison, failure analysis |
| Hamming AI | Voice-specific | Custom LLM-as-judge scorers, production monitoring |

### Evaluation Dimensions for Voice Agents
Based on industry practice (Hamming AI, Retell AI, Braintrust), voice agent evaluation typically covers:
- **Task Completion**: Did the agent achieve the call objective?
- **Natural Flow**: Conversation naturalness, turn-taking quality
- **Error Handling**: Recovery from ASR errors, clarification requests
- **Compliance**: Adherence to scripts/prompts
- **Accuracy**: Factual correctness of information provided
- **Caller Experience**: Overall sentiment, politeness, cultural appropriateness

---

## 4. Voice Agent Evaluation Platforms (2025)

### VoiceAgentBench (arxiv:2510.07978)
*"VoiceAgentBench: Are Voice Assistants Ready for Agentic Tasks?"* — a dedicated benchmark for evaluating voice assistants on agentic tasks, going beyond simple ASR accuracy.

### Hamming AI
Enterprise voice agent testing platform. Features:
- Auto-generated test scenarios
- Production call replay
- 50+ built-in metrics
- Custom LLM-as-judge scorers for business-specific behaviors
- Real-time production monitoring with configurable alerts
- Cross-call pattern detection

### Braintrust
Creates testing agents that simulate real callers (interruptions, frustration, topic changes). Handles evaluation with datasets, scorers, cross-run comparison.

### AIEWF-Eval (Daily.co)
Open-source benchmark for LLMs in voice agent use cases. Tests:
- Latency
- Tool calling accuracy
- Instruction following
- Knowledge grounding in long multi-turn conversations

### Soniox Benchmarks (2025)
Cross-language CER/WER benchmarks across multiple speech recognition providers, providing standardized comparison data.

---

## 5. Auto-Research Frameworks

### STORM (Stanford)
LLM-powered knowledge curation system. Core innovation: **Perspective-Guided Question Asking** — discovers different perspectives by surveying existing articles on similar topics, then uses them to control question generation. Generates full-length reports with citations.

- **GitHub**: github.com/stanford-oval/storm (5k+ stars)
- **Architecture**: Plan perspectives → Interview simulation → Outline → Article generation → Polish

### GPT-Researcher
Autonomous agent for deep research using any LLM. Inspired by STORM, uses a team of AI agents from planning to publication. Average run generates 5-6 page reports in multiple formats.

- **GitHub**: github.com/assafelovic/gpt-researcher (15k+ stars)
- **Architecture**: Plan → Divide into subtasks → Execute subtasks → Synthesize

### AI-Researcher (NeurIPS 2025)
*"AI-Researcher: Autonomous Scientific Innovation"* — takes a list of papers and autonomously generates research insights. Production version at novix.science.

---

## 6. Recommendations for japanese-eval-V1

### Immediate (Implemented)
1. **CER as primary metric** with lenient CER option (kanji/hiragana/katakana equivalence) ✅
2. **Hallucination detection** via kanji-in-hypothesis-not-in-reference ✅
3. **YAML-driven benchmark tasks** following EleutherAI patterns ✅
4. **LLM-as-judge with 6 Japanese business dimensions** ✅
5. **Auto-research pipeline** (Plan → Search → Read → Synthesize) ✅

### Next Steps
1. **Add ReazonSpeech and JSUT as standard evaluation sets** — enables cross-model comparison on established benchmarks
2. **Integrate kotoba-whisper** as a provider — 6.3x faster than whisper-large-v3 with comparable CER
3. **Add web search source** to research pipeline — ArXiv API is unreliable, web search provides broader coverage
4. **Implement VoiceAgentBench-style agentic evaluation** — test voice agents on task completion, not just ASR accuracy
5. **Production monitoring integration** — inspired by Hamming AI's cross-call pattern detection
6. **STORM-style perspective-guided questioning** in research planner — improves question diversity and depth

---

## References

1. "Lenient Evaluation of Japanese Speech Recognition: Modeling Naturally Occurring Spelling Inconsistency" (ACL 2023). arxiv:2306.04530
2. "Benchmarking Japanese Speech Recognition on ASR-LLM Setups with Multi-Pass Augmented Generative Error Correction" (2024). arxiv:2408.16180
3. "Advocating Character Error Rate for Multilingual ASR Evaluation" (NAACL 2025 Findings). aclanthology.org/2025.findings-naacl.277
4. "Efficient Adaptation of Multilingual Models for Japanese ASR" (2024). arxiv:2412.10705
5. ReazonSpeech: A Free and Massive Corpus for Japanese ASR. research.reazon.jp
6. kotoba-whisper v1.0 / v2.0. huggingface.co/kotoba-tech
7. "Learning an Efficient Multi-Turn Dialogue Evaluator from Multiple LLM Judges" (2025). arxiv:2508.00454
8. "When AIs Judge AIs: The Rise of Agent-as-a-Judge Evaluation" (2025). arxiv:2508.02994
9. "Evaluating and Mitigating LLM-as-a-judge Bias in Communication Systems" (2025). arxiv:2510.12462
10. "VoiceAgentBench: Are Voice Assistants Ready for Agentic Tasks?" (2025). arxiv:2510.07978
11. STORM: Knowledge Curation System. github.com/stanford-oval/storm
12. GPT-Researcher: Autonomous Research Agent. github.com/assafelovic/gpt-researcher
13. AI-Researcher: Autonomous Scientific Innovation (NeurIPS 2025). github.com/HKUDS/AI-Researcher
14. EleutherAI LM Evaluation Harness. github.com/EleutherAI/lm-evaluation-harness
15. japanese-asr HuggingFace Organization. huggingface.co/japanese-asr
16. Hamming AI Voice Agent Testing. hamming.ai
17. Soniox Speech-to-Text Benchmarks 2025. soniox.com/benchmarks
