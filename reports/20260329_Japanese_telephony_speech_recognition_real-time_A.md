# Japanese Telephony Speech Recognition: Real-time ASR, Noise Robustness, and Keigo Detection

## Executive Summary

This report analyzes the current landscape of Japanese telephony speech recognition, focusing on real-time ASR capabilities, noise robustness, and honorific speech (keigo) detection. The research reveals a growing ecosystem of specialized Japanese ASR models, primarily built on Whisper foundations and trained on the ReazonSpeech corpus, though significant gaps remain in telephony-specific optimization and keigo classification systems.

## Key Findings

### Real-time ASR Models for Japanese Telephony

**Current State**: The available research findings show limited evidence of dedicated real-time Japanese telephony ASR systems. The most relevant models identified are:

- **Distil-Whisper Japanese variants**: Multiple distilled versions of Whisper optimized for Japanese, including `distil-whisper-large-v3-ja-reazonspeech-all` [1] and smaller variants trained on ReazonSpeech datasets
- **Performance metrics gap**: No specific latency or accuracy metrics for telephony conditions were found in the available resources
- **Real-time optimization**: The distilled models suggest attempts at computational efficiency, but telephony-specific benchmarks are absent

### Noise Robustness in Japanese ASR

**Limited telephony-specific solutions**: The research identified only one noise-robust system:
- **WhisperJAV** [2]: Explicitly designed for noise-robust ASR using Qwen3-ASR, local LLM, Whisper, and TEN-VAD, though not specifically for telephony applications

**Research gap**: No comprehensive studies on channel distortion effects, background noise impact, or telephony-specific acoustic modeling for Japanese were found in the available resources.

### Keigo Detection and Classification

**Critical research gap**: The findings reveal a significant absence of computational approaches for automatic keigo detection in Japanese conversational AI. No dedicated models, datasets, or evaluation frameworks for honorific speech classification were identified in the available resources.

**Linguistic feature analysis**: No systematic studies on effective linguistic features for keigo detection were found, representing a major limitation for customer service applications.

### Japanese ASR Evaluation Datasets

**Primary corpus**: ReazonSpeech emerges as the dominant training corpus [3], with multiple derivative datasets:
- `whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized` (8,823 downloads)
- `whisper_transcriptions.reazon_speech_all` (2,443 downloads)
- Various conversational speech recognition corpora from DataoceanAI with minimal adoption (4-6 downloads each)

**Telephony coverage limitation**: No datasets specifically designed for telephony conditions or keigo coverage were identified.

### Customer Service Application Challenges

**Identified limitations**:
1. **Lack of keigo understanding**: No available systems for politeness level detection
2. **Real-time processing gaps**: Limited evidence of sub-200ms latency optimization
3. **Telephony adaptation**: Absence of channel-specific acoustic models
4. **Integration complexity**: No end-to-end customer service frameworks identified

## Notable Papers

*Note: The research findings provided do not include academic papers. This represents a significant gap in the literature review.*

## Notable Tools & Models

### ASR Models
- **japanese-asr/distil-whisper-large-v3-ja-reazonspeech-all** [1] (24 downloads) - Distilled Whisper for Japanese ASR
- **japanese-asr/distil-whisper-bilingual-v1.0** [4] (19 downloads) - Bilingual Japanese-English ASR
- **NadiaHolmlund/ASR_Japanese_Fine_Tuned_Whisper_Model** [5] (10 downloads) - Fine-tuned Whisper for Japanese

### TTS Systems  
- **myshell-ai/MeloTTS** [6] (7,301 GitHub stars) - Multi-lingual TTS including Japanese
- **atomicoo/FCH-TTS** [7] (281 GitHub stars) - Fast multilingual TTS with Japanese support
- **japanese-parler-tts-mini** [8] (1,082 downloads) - Compact Japanese TTS model

### Specialized Tools
- **reazon-research/ReazonSpeech** [3] (376 GitHub stars) - Massive Japanese speech corpus
- **meizhong986/WhisperJAV** [2] (1,338 GitHub stars) - Noise-robust ASR system
- **j3soon/whisper-to-input** [9] (116 GitHub stars) - Android multilingual speech input

## Available Datasets

### ASR Training Data
1. **ReazonSpeech derivatives** - Multiple vectorized transcription datasets (2,333-8,823 downloads)
2. **japanese-asr/whisper_transcriptions.mls** [10] (3,405 downloads) - Multilingual LibriSpeech Japanese subset
3. **UniDataPro/japanese-speech-recognition-dataset** [11] (20 downloads) - General Japanese ASR corpus

### TTS Resources
1. **QWEN3-TTS voice datasets** [12] - Emotional and designed voice collections (377-7,137 downloads)
2. **japanese-singing-voice** [13] (265 downloads) - Specialized singing voice corpus

## Recommendations for Japanese Voice AI Evaluation

### Immediate Priorities
1. **Develop telephony-specific benchmarks**: Create evaluation datasets that include channel distortion, codec effects, and typical telephony noise conditions
2. **Establish keigo detection frameworks**: Develop computational models and evaluation metrics for honorific speech classification
3. **Real-time performance standardization**: Define latency and accuracy benchmarks specifically for Japanese telephony ASR

### Technical Development Needs
1. **Noise robustness enhancement**: Extend successful approaches like WhisperJAV to telephony-specific conditions
2. **Customer service integration**: Develop end-to-end frameworks combining ASR, keigo detection, and response generation
3. **Multilingual capability**: Leverage bilingual models for international customer service scenarios

### Research Gaps to Address
1. **Politeness level understanding**: Systematic study of linguistic features for keigo classification
2. **Domain adaptation**: Telephony-specific acoustic model training and evaluation
3. **Cultural context integration**: Incorporation of Japanese business communication norms into AI systems

## References

1. Hugging Face. japanese-asr/distil-whisper-large-v3-ja-reazonspeech-all. https://huggingface.co/japanese-asr/distil-whisper-large-v3-ja-reazonspeech-all
2. GitHub. meizhong986/WhisperJAV. https://github.com/meizhong986/WhisperJAV
3. GitHub. reazon-research/ReazonSpeech. https://github.com/reazon-research/ReazonSpeech  
4. Hugging Face. japanese-asr/distil-whisper-bilingual-v1.0. https://huggingface.co/japanese-asr/distil-whisper-bilingual-v1.0
5. Hugging Face. NadiaHolmlund/ASR_Japanese_Fine_Tuned_Whisper_Model. https://huggingface.co/NadiaHolmlund/ASR_Japanese_Fine_Tuned_Whisper_Model
6. GitHub. myshell-ai/MeloTTS. https://github.com/myshell-ai/MeloTTS
7. GitHub. atomicoo/FCH-TTS. https://github.com/atomicoo/FCH-TTS
8. Hugging Face. 2121-8/japanese-parler-tts-mini. https://huggingface.co/2121-8/japanese-parler-tts-mini
9. GitHub. j3soon/whisper-to-input. https://github.com/j3soon/whisper-to-input
10. Hugging Face. japanese-asr/whisper_transcriptions.mls. https://huggingface.co/datasets/japanese-asr/whisper_transcriptions.mls
11. Hugging Face. UniDataPro/japanese-speech-recognition-dataset. https://huggingface.co/datasets/UniDataPro/japanese-speech-recognition-dataset
12. Hugging Face. Akjava/QWEN3-TTS-Voice-Clone-100-Japanese-Female-ITA-Corpus-Emotion. https://huggingface.co/datasets/Akjava/QWEN3-TTS-Voice-Clone-100-Japanese-Female-ITA-Corpus-Emotion
13. Hugging Face. tts-dataset/japanese-singing-voice. https://huggingface.co/datasets/tts-dataset/japanese-singing-voice