# LLM-as-Judge for Voice Agent Conversation Quality: Automated Scoring and Multi-Dimensional Rubrics

## Executive Summary

This report analyzes the current state of LLM-as-judge frameworks for evaluating Japanese voice agent conversation quality. While substantial resources exist for Japanese speech recognition and text-to-speech synthesis, significant gaps remain in developing comprehensive automated evaluation systems that address Japanese linguistic nuances and multi-dimensional conversation quality assessment.

## Key Findings

### 1. LLM-as-Judge Framework Adaptation for Japanese Voice Agents

**Current State**: Limited specialized frameworks exist for Japanese conversational AI evaluation with linguistic nuance awareness.

**Available Resources**:
- The **ai-debate-analyzer** system [1] provides a foundational framework for conversation analysis using LLMs with speaker performance evaluation and sentiment analysis capabilities
- **Evalverse-Complete** [2] offers a modular LLM-powered assessment framework supporting voice and text evaluation, though Japanese-specific adaptations are unclear
- No identified frameworks specifically address keigo (honorific language) evaluation or context-dependent expression assessment

**Gap**: Dedicated LLM-as-judge systems for Japanese linguistic nuances (keigo, context-dependent expressions) are not available in current open-source implementations.

### 2. Multi-Dimensional Rubrics for Japanese Voice Conversation Scoring

**Current State**: Existing evaluation systems focus primarily on technical metrics (ASR accuracy, TTS quality) rather than comprehensive conversation quality assessment.

**Available Components**:
- **Japanese emotion recognition**: wav2vec2-xlsr model [3] provides audio-based emotion classification (343 downloads)
- **Interactive analysis tools**: Fudoki [4] enables real-time Japanese text analysis and speech synthesis evaluation
- **Debate analysis framework**: Comprehensive scoring system including argument quality, sentiment analysis, and detailed reasoning [1]

**Gap**: No identified multi-dimensional rubrics specifically designed for Japanese conversational AI that integrate fluency, naturalness, appropriateness, and task completion metrics.

### 3. LLM Judge Performance for Japanese Speech Recognition and TTS Evaluation

**ASR Performance Evaluation**:
- **ReazonSpeech-based models** [5,6,7]: Multiple distilled Whisper variants specifically trained on Japanese speech data
- **Bilingual capabilities**: distil-whisper-bilingual-v1.0 [8] supports Japanese-English mixed evaluation (19 downloads)
- **Evaluation datasets**: Comprehensive Whisper transcription datasets with WER metrics available [9,10,11]

**TTS Quality Assessment**:
- **Japanese TTS models**: Multiple implementations including japanese-parler-tts-mini [12] (1,082 downloads) and japanese_speecht5_tts [13] (373 downloads)
- **Voice design datasets**: QWEN3-TTS voice datasets [14,15] provide evaluation baselines for Japanese female voices

**Performance Metrics**: Current evaluation focuses on Character Error Rate (CER) and Word Error Rate (WER) metrics, with limited LLM-based qualitative assessment frameworks.

### 4. Code-Switching Evaluation Challenges

**Current Capabilities**:
- **Bilingual ASR support**: Limited bilingual models available (distil-whisper-bilingual-v1.0) [8]
- **Mixed-language datasets**: Japanese-English conversational speech recognition corpora [16] (4 downloads)
- **Multi-language TTS**: Some systems support multiple languages including Japanese [17]

**Major Gaps**: 
- No specialized evaluation frameworks for Japanese-English code-switching quality assessment
- Limited training data for mixed-language conversation evaluation
- Absence of LLM judges specifically designed for code-switching appropriateness

### 5. Ground Truth Dataset Development

**Available Datasets**:
- **Large-scale ASR datasets**: ReazonSpeech transcriptions [9,10,11] with thousands of downloads
- **Conversational speech corpora**: Japanese conversational speech recognition datasets [18] (6 downloads)
- **TTS evaluation data**: Japanese voice datasets with emotional annotations [14] (7,137 downloads)
- **Singing voice data**: Specialized Japanese singing voice datasets [19] (265 downloads)

**Quality Metrics**: Existing datasets provide WER-filtered transcriptions (WER 10.0 threshold) and vectorized representations for evaluation purposes.

**Gaps**: Limited ground truth datasets specifically designed for LLM judge training and validation in conversational quality assessment.

## Notable Papers

*Note: No specific academic papers were identified in the provided research findings. This represents a significant gap in the literature for Japanese voice AI evaluation methodologies.*

## Notable Tools & Models

### Speech Recognition Models
1. **japanese-asr/distil-whisper-large-v3-ja-reazonspeech-all** (24 downloads) - Japanese-optimized Whisper variant [5]
2. **japanese-asr/distil-whisper-bilingual-v1.0** (19 downloads) - Bilingual Japanese-English ASR [8]
3. **NadiaHolmlund/ASR_Japanese_Fine_Tuned_Whisper_Model** (10 downloads) - Fine-tuned Japanese Whisper [20]

### Text-to-Speech Models
1. **2121-8/japanese-parler-tts-mini** (1,082 downloads) - Compact Japanese TTS system [12]
2. **esnya/japanese_speecht5_tts** (373 downloads) - SpeechT5-based Japanese TTS [13]

### Analysis and Evaluation Tools
1. **MeloTTS** (7,301 stars) - Multi-lingual TTS with Japanese support [21]
2. **Fudoki** (517 stars) - Interactive Japanese text analysis and speech synthesis [4]
3. **AivisSpeech** (424 stars) - AI voice imitation system [22]

### Emotion and Quality Assessment
1. **wav2vec2-xlsr-japanese-speech-emotion-recognition** (343 downloads) - Japanese emotion recognition [3]

## Available Datasets

### ASR Evaluation Datasets
1. **whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized** (8,823 downloads) [9]
2. **whisper_transcriptions.mls** (3,405 downloads) [10]
3. **whisper_transcriptions.reazon_speech_all** (2,443 downloads) [11]

### TTS and Voice Datasets
1. **QWEN3-TTS-Voice-Clone-100-Japanese-Female-ITA-Corpus-Emotion** (7,137 downloads) [14]
2. **QWEN3-TTS-Voice-Design-100-Japanese-Female-Designed-Voices** (755 downloads) [15]
3. **japanese-singing-voice** (265 downloads) [19]

### Conversational Speech Datasets
1. **Japanese-Conversational-Speech-Recognition-Corpus** (6 downloads) [18]
2. **Japanese_English_Speech_Recognition_Corpus_Conversations** (4 downloads) [16]

## Recommendations for Japanese Voice AI Evaluation

### Immediate Actions
1. **Develop Japanese-specific LLM judges** trained on linguistic nuances including keigo and context-dependent expressions
2. **Create comprehensive evaluation rubrics** that integrate technical metrics (CER/WER) with conversational quality assessments
3. **Establish code-switching evaluation protocols** for Japanese-English mixed conversations

### Medium-term Development
1. **Build large-scale ground truth datasets** specifically for LLM judge training with human-annotated conversation quality scores
2. **Implement multi-dimensional scoring systems** covering fluency, naturalness, appropriateness, and task completion
3. **Develop telephony-specific evaluation frameworks** addressing quality degradation in voice communication contexts

### Long-term Research Priorities
1. **Cross-cultural appropriateness evaluation** for Japanese business and social contexts
2. **Real-time conversation quality monitoring** systems for production voice agents
3. **Standardized benchmarks** for Japanese voice AI evaluation across different domains and use cases

## References

1. amrut20562/ai-debate-analyzer. GitHub. https://github.com/amrut20562/ai-debate-analyzer
2. SuryaAnything/Evalverse-Complete. GitHub. https://github.com/SuryaAnything/Evalverse-Complete
3. Bagus/wav2vec2-xlsr-japanese-speech-emotion-recognition. HuggingFace. https://huggingface.co/Bagus/wav2vec2-xlsr-japanese-speech-emotion-recognition
4. iamcheyan/fudoki. GitHub. https://github.com/iamcheyan/fudoki
5. japanese-asr/distil-whisper-large-v3-ja-reazonspeech-all. HuggingFace. https://huggingface.co/japanese-asr/distil-whisper-large-v3-ja-reazonspeech-all
6. japanese-asr/distil-whisper-large-v3-ja-reazonspeech-large. HuggingFace. https://huggingface.co/japanese-asr/distil-whisper-large-v3-ja-reazonspeech-large
7. japanese-asr/distil-whisper-large-v3-ja-reazonspeech-small. HuggingFace. https://huggingface.co/japanese-asr/distil-whisper-large-v3-ja-reazonspeech-small
8. japanese-asr/distil-whisper-bilingual-v1.0. HuggingFace. https://huggingface.co/japanese-asr/distil-whisper-bilingual-v1.0
9. japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized. HuggingFace Datasets. https://huggingface.co/datasets/japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized
10. japanese-asr/whisper_transcriptions.mls. HuggingFace Datasets. https://huggingface.co/datasets/japanese-asr/whisper_transcriptions.mls
11. japanese-asr/whisper_transcriptions.reazon_speech_all. HuggingFace Datasets. https://huggingface.co/datasets/japanese-asr/whisper_transcriptions.reazon_speech_all
12. 2121-8/japanese-parler-tts-mini. HuggingFace. https://huggingface.co/2121-8/japanese-parler-tts-mini
13. esnya/japanese_speecht5_tts. HuggingFace. https://huggingface.co/esnya/japanese_speecht5_tts
14. Akjava/QWEN3-TTS-Voice-Clone-100-Japanese-Female-ITA-Corpus-Emotion. HuggingFace Datasets. https://huggingface.co/datasets/Akjava/QWEN3-TTS-Voice-Clone-100-Japanese-Female-ITA-Corpus-Emotion
15. Akjava/QWEN3-TTS-Voice-Design-100-Japanese-Female-Designed-Voices. HuggingFace Datasets. https://huggingface.co/datasets/Akjava/QWEN3-TTS-Voice-Design-100-Japanese-Female-Designed-Voices
16. DataoceanAI/Japanese_English_Speech_Recognition_Corpus_Conversations. HuggingFace Datasets. https://huggingface.co/datasets/DataoceanAI/Japanese_English_Speech_Recognition_Corpus_Conversations
17. atomicoo/FCH-TTS. GitHub. https://github.com/atomicoo/FCH-TTS
18. DataoceanAI/Dolphin_Model_Japanese-Conversational-Speech-Recognition-Corpus. HuggingFace Datasets. https://huggingface.co/datasets/DataoceanAI/Dolphin_Model_Japanese-Conversational-Speech-Recognition-Corpus
19. tts-dataset/japanese-singing-voice. HuggingFace Datasets. https://huggingface.co/datasets/tts-dataset/japanese-singing-voice
20. NadiaHolmlund/ASR_Japanese_Fine_Tuned_Whisper_Model. HuggingFace. https://huggingface.co/NadiaHolmlund/ASR_Japanese_Fine_Tuned_Whisper_Model
21. myshell-ai/MeloTTS. GitHub. https://github.com/myshell-ai/MeloTTS
22. Aivis-Project/AivisSpeech. GitHub. https://github.com/Aivis-Project/AivisSpeech