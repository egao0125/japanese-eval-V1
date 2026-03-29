# Japanese TTS Evaluation: Naturalness, Prosody, and MOS Scoring for Business Telephony

## Executive Summary

This report analyzes current approaches to evaluating Japanese text-to-speech (TTS) systems in business telephony applications, focusing on naturalness, prosodic features, and Mean Opinion Score (MOS) methodologies. While significant progress has been made in Japanese TTS development with models like MeloTTS and specialized datasets like ReazonSpeech, substantial gaps remain in standardized evaluation frameworks specifically designed for telephony environments and business use cases.

## Key Findings

### 1. MOS Scoring Methodologies for Japanese TTS in Business Telephony

**Research Gap Identified**: The available resources show limited specific methodologies for MOS scoring in business telephony contexts. Current Japanese TTS models primarily focus on general-purpose evaluation rather than telephony-specific quality metrics.

**Available Resources**:
- MeloTTS provides multi-lingual TTS capabilities including Japanese but lacks telephony-specific evaluation protocols [1]
- Japanese-specific TTS models like `esnya/japanese_speecht5_tts` (373 downloads) and `2121-8/japanese-parler-tts-mini` (1,082 downloads) exist but without documented telephony evaluation frameworks [2,3]

### 2. Prosodic Features Impact on Japanese Voice Agents

**Key Observations**:
- Japanese-specific prosodic modeling is addressed in specialized models like `Bagus/wav2vec2-xlsr-japanese-speech-emotion-recognition` (343 downloads), which focuses on emotion recognition in Japanese speech [4]
- The ReazonSpeech corpus provides a massive Japanese speech dataset that could support prosodic analysis, though specific telephony prosodic evaluation metrics are not documented [5]

**Research Gap**: Limited documentation on how prosodic features specifically impact user acceptance in telephone customer service scenarios.

### 3. Evaluation Metrics: Telephony vs. General-Purpose Applications

**Identified Differences**:
- General-purpose models dominate the available resources, with limited telephony-specific evaluation frameworks
- Available datasets like `japanese-asr/whisper_transcriptions.reazon_speech_all` (2,443 downloads) focus on general speech recognition rather than telephony-specific quality metrics [6]

**Research Gap**: Insufficient research on bandwidth limitations, compression artifacts, and noise robustness specific to telephony environments in Japanese TTS evaluation.

### 4. Japanese-Specific Linguistic Features in TTS Evaluation

**Key Resources Identified**:
- Fudoki provides interactive Japanese text analysis and speech synthesis capabilities, potentially supporting pitch accent and mora timing analysis [7]
- Multiple distilled Whisper models specifically trained for Japanese (e.g., `japanese-asr/distil-whisper-large-v3-ja-reazonspeech-all` with 24 downloads) suggest attention to Japanese linguistic specificity [8]

**Observed Capabilities**:
- FCH-TTS explicitly supports Japanese among multiple languages, indicating consideration of language-specific features [9]
- VITS-based Japanese models like `AlexandaJerry/whisper-vits-japanese` suggest specialized architectures for Japanese speech synthesis [10]

### 5. Benchmarks and Datasets for Real-World Telephony Evaluation

**Primary Resources**:
- **ReazonSpeech**: Massive open Japanese speech corpus, fundamental for Japanese voice AI evaluation [5]
- **Whisper Transcription Datasets**: Multiple vectorized datasets with WER metrics (e.g., `japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized` with 8,823 downloads) [11]
- **Specialized TTS Datasets**: Emotion-based datasets like `Akjava/QWEN3-TTS-Voice-Clone-100-Japanese-Female-ITA-Corpus-Emotion` (7,137 downloads) [12]

**Research Gap**: No identified datasets specifically designed for telephony environments or business customer service scenarios.

## Notable Papers

*Note: The provided research findings primarily contain GitHub repositories, HuggingFace models, and datasets rather than peer-reviewed papers. This represents a significant gap in academic literature specifically addressing Japanese TTS evaluation in business telephony contexts.*

## Notable Tools & Models

### Open-Source TTS Libraries
1. **MeloTTS** - High-quality multi-lingual TTS with Japanese support (7,301 stars) [1]
2. **FCH-TTS** - Fast multilingual TTS including Japanese (281 stars) [9]

### Specialized Japanese Models
1. **japanese-asr/distil-whisper-large-v3-ja-reazonspeech-all** (24 downloads) - ASR evaluation [8]
2. **esnya/japanese_speecht5_tts** (373 downloads) - Japanese TTS [2]
3. **2121-8/japanese-parler-tts-mini** (1,082 downloads) - Compact Japanese TTS [3]
4. **Bagus/wav2vec2-xlsr-japanese-speech-emotion-recognition** (343 downloads) - Emotion analysis [4]

### Analysis Tools
1. **Fudoki** - Interactive Japanese text analysis and speech synthesis (517 stars) [7]

## Available Datasets

### Primary Speech Corpora
1. **ReazonSpeech** - Massive open Japanese speech corpus [5]
2. **japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized** (8,823 downloads) [11]
3. **japanese-asr/whisper_transcriptions.mls** (3,405 downloads) [13]

### Specialized TTS Datasets
1. **Akjava/QWEN3-TTS-Voice-Clone-100-Japanese-Female-ITA-Corpus-Emotion** (7,137 downloads) [12]
2. **tts-dataset/japanese-singing-voice** (265 downloads) [14]
3. **Akjava/QWEN3-TTS-Voice-Design-100-Japanese-Female-Designed-Voices** (755 downloads) [15]

## Recommendations for Japanese Voice AI Evaluation

### Immediate Actions
1. **Establish Telephony-Specific Evaluation Protocols**: Develop MOS scoring methodologies that account for telephony bandwidth limitations, compression artifacts, and noise conditions specific to business phone systems.

2. **Create Business Telephony Datasets**: Build datasets that reflect real-world customer service scenarios, including background noise, emotional speech patterns, and domain-specific vocabulary.

3. **Develop Japanese-Specific Prosodic Metrics**: Establish evaluation frameworks that specifically measure pitch accent accuracy, mora timing, and intonation patterns crucial for natural-sounding Japanese speech.

### Research Priorities
1. **Comparative Analysis**: Conduct systematic comparisons between general-purpose and telephony-specific evaluation metrics for Japanese TTS systems.

2. **User Acceptance Studies**: Perform longitudinal studies measuring customer satisfaction and task completion rates with different Japanese TTS systems in actual business telephony environments.

3. **Linguistic Feature Impact Assessment**: Quantify how Japanese-specific features (pitch accent, mora timing) affect perceived naturalness in telephony vs. general applications.

### Technical Implementation
1. **Leverage Existing Resources**: Utilize ReazonSpeech corpus as a foundation while augmenting with telephony-specific recordings.

2. **Multi-Model Evaluation**: Establish benchmarks using available models (MeloTTS, FCH-TTS, Japanese SpeechT5) across consistent telephony-specific test scenarios.

3. **Real-World Deployment Testing**: Implement A/B testing frameworks for Japanese TTS systems in actual business telephony environments.

## References

[1] myshell-ai/MeloTTS. GitHub. https://github.com/myshell-ai/MeloTTS

[2] esnya/japanese_speecht5_tts. HuggingFace. https://huggingface.co/esnya/japanese_speecht5_tts

[3] 2121-8/japanese-parler-tts-mini. HuggingFace. https://huggingface.co/2121-8/japanese-parler-tts-mini

[4] Bagus/wav2vec2-xlsr-japanese-speech-emotion-recognition. HuggingFace. https://huggingface.co/Bagus/wav2vec2-xlsr-japanese-speech-emotion-recognition

[5] reazon-research/ReazonSpeech. GitHub. https://github.com/reazon-research/ReazonSpeech

[6] japanese-asr/whisper_transcriptions.reazon_speech_all. HuggingFace. https://huggingface.co/datasets/japanese-asr/whisper_transcriptions.reazon_speech_all

[7] iamcheyan/fudoki. GitHub. https://github.com/iamcheyan/fudoki

[8] japanese-asr/distil-whisper-large-v3-ja-reazonspeech-all. HuggingFace. https://huggingface.co/japanese-asr/distil-whisper-large-v3-ja-reazonspeech-all

[9] atomicoo/FCH-TTS. GitHub. https://github.com/atomicoo/FCH-TTS

[10] AlexandaJerry/whisper-vits-japanese. GitHub. https://github.com/AlexandaJerry/whisper-vits-japanese

[11] japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized. HuggingFace. https://huggingface.co/datasets/japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized

[12] Akjava/QWEN3-TTS-Voice-Clone-100-Japanese-Female-ITA-Corpus-Emotion. HuggingFace. https://huggingface.co/datasets/Akjava/QWEN3-TTS-Voice-Clone-100-Japanese-Female-ITA-Corpus-Emotion

[13] japanese-asr/whisper_transcriptions.mls. HuggingFace. https://huggingface.co/datasets/japanese-asr/whisper_transcriptions.mls

[14] tts-dataset/japanese-singing-voice. HuggingFace. https://huggingface.co/datasets/tts-dataset/japanese-singing-voice

[15] Akjava/QWEN3-TTS-Voice-Design-100-Japanese-Female-Designed-Voices. HuggingFace. https://huggingface.co/datasets/Akjava/QWEN3-TTS-Voice-Design-100-Japanese-Female-Designed-Voices