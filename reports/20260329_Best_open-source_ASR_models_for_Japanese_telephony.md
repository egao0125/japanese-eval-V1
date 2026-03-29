# Best Open-Source ASR Models for Japanese Telephony Speech Recognition in 2025-2026

## Executive Summary

This report examines the landscape of open-source Japanese automatic speech recognition (ASR) models optimized for telephony applications, focusing on 8kHz noisy conversational audio. While significant progress has been made in Japanese ASR with specialized models like distilled Whisper variants trained on ReazonSpeech data and dedicated Japanese TTS systems, there remains a notable research gap specifically addressing telephony-grade (8kHz) audio processing and comprehensive comparative analysis between different architectural approaches.

## Key Findings

### Model Architecture Performance on Japanese 8kHz Telephony Audio

**Research Gap Identified**: The available research findings lack comprehensive benchmarking data comparing Whisper variants, Kotoba-Whisper, SenseVoice, and ReazonSpeech architectures specifically on 8kHz telephony audio. However, several promising developments emerge:

- **Distilled Whisper Models**: The japanese-asr organization has developed multiple distilled Whisper variants (distil-whisper-large-v3-ja-reazonspeech-all, distil-whisper-large-v3-ja-reazonspeech-large, distil-whisper-large-v3-ja-reazonspeech-small) [1,2,3] that offer improved inference speed while maintaining accuracy for Japanese ASR.
- **ReazonSpeech Integration**: Multiple models leverage the ReazonSpeech corpus [4], indicating its importance as a training foundation for Japanese ASR systems.

### CTC vs. Transducer vs. Transformer Architectures

**Insufficient Data**: The research findings do not provide specific comparative analysis between CTC, transducer, and transformer architectures for Japanese telephony speech recognition. This represents a critical research gap requiring further investigation.

### Fine-tuning Strategies and Data Augmentation

**Limited Coverage**: While specialized Japanese ASR models exist, specific fine-tuning strategies for 8kHz telephony conditions are not well-documented in the available findings. The WhisperJAV project [5] demonstrates noise-robust ASR capabilities but focuses on different use cases.

### Model Size and Inference Speed Trade-offs

The distilled Whisper variants provide evidence of size-speed optimization:
- **Small variant**: japanese-asr/distil-whisper-large-v3-ja-reazonspeech-small (5 downloads) [3]
- **Large variant**: japanese-asr/distil-whisper-large-v3-ja-reazonspeech-large (6 downloads) [2]
- **All variant**: distil-whisper-large-v3-ja-reazonspeech-all (24 downloads) [1]

The download statistics suggest moderate adoption, with the comprehensive "all" variant being most popular.

### Evaluation Methodologies and Benchmark Datasets

**Promising Resources Identified**:
- ReazonSpeech corpus provides massive open Japanese speech data [4]
- Vectorized datasets with WER 10.0 filtering available [6,7]
- Conversational speech datasets from DataoceanAI [8,9]

## Notable Papers

*Research Gap*: The provided findings do not include specific academic papers addressing Japanese telephony ASR. This indicates a need for more comprehensive literature review focusing on telephony-specific Japanese speech recognition research.

## Notable Tools & Models

### Primary ASR Models
1. **japanese-asr/distil-whisper-large-v3-ja-reazonspeech-all** - 24 downloads [1]
2. **japanese-asr/distil-whisper-bilingual-v1.0** - 19 downloads [10]
3. **NadiaHolmlund/ASR_Japanese_Fine_Tuned_Whisper_Model** - 10 downloads [11]
4. **reazon-research/ReazonSpeech** - 376 GitHub stars [4]

### Supporting Tools
1. **myshell-ai/MeloTTS** - 7,302 GitHub stars, multi-lingual TTS including Japanese [12]
2. **Ayanaminn/N46Whisper** - 1,705 GitHub stars, Whisper-based Japanese subtitle generator [13]
3. **meizhong986/WhisperJAV** - 1,338 GitHub stars, noise-robust ASR with TEN-VAD [5]

## Available Datasets

### ASR-Specific Datasets
1. **japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized** - 8,823 downloads [6]
2. **japanese-asr/whisper_transcriptions.mls** - 3,405 downloads [14]
3. **DataoceanAI/Dolphin_Model_Japanese-Conversational-Speech-Recognition-Corpus** - 6 downloads [8]
4. **DataoceanAI/Japanese_English_Speech_Recognition_Corpus_Conversations** - 4 downloads [9]

### TTS Datasets
1. **Akjava/QWEN3-TTS-Voice-Clone-100-Japanese-Female-ITA-Corpus-Emotion** - 7,137 downloads [15]
2. **tts-dataset/japanese-singing-voice** - 265 downloads [16]

## Recommendations for Japanese Voice AI Evaluation

### Immediate Actions
1. **Establish Telephony-Specific Benchmarks**: Develop standardized 8kHz Japanese telephony speech datasets with varying noise conditions and speaker demographics.

2. **Comprehensive Architecture Comparison**: Conduct systematic evaluation comparing Whisper variants, ReazonSpeech, SenseVoice, and CTC/transducer architectures on identical telephony datasets.

3. **Fine-tuning Protocol Development**: Create standardized fine-tuning procedures specifically for 8kHz telephony adaptation, incorporating data augmentation techniques for noise robustness.

### Long-term Research Priorities
1. **Real-time Performance Optimization**: Balance accuracy, model size, and inference speed for practical telephony deployment.

2. **Multi-domain Evaluation Framework**: Establish comprehensive evaluation methodology incorporating CER, WER, and perceptual quality metrics across diverse telephony scenarios.

3. **Open-source Collaboration**: Leverage the ReazonSpeech foundation to build community-driven telephony ASR benchmarks.

## References

[1] https://huggingface.co/japanese-asr/distil-whisper-large-v3-ja-reazonspeech-all
[2] https://huggingface.co/japanese-asr/distil-whisper-large-v3-ja-reazonspeech-large
[3] https://huggingface.co/japanese-asr/distil-whisper-large-v3-ja-reazonspeech-small
[4] https://github.com/reazon-research/ReazonSpeech
[5] https://github.com/meizhong986/WhisperJAV
[6] https://huggingface.co/datasets/japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized
[7] https://huggingface.co/datasets/japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized
[8] https://huggingface.co/datasets/DataoceanAI/Dolphin_Model_Japanese-Conversational-Speech-Recognition-Corpus
[9] https://huggingface.co/datasets/DataoceanAI/Japanese_English_Speech_Recognition_Corpus_Conversations
[10] https://huggingface.co/japanese-asr/distil-whisper-bilingual-v1.0
[11] https://huggingface.co/NadiaHolmlund/ASR_Japanese_Fine_Tuned_Whisper_Model
[12] https://github.com/myshell-ai/MeloTTS
[13] https://github.com/Ayanaminn/N46Whisper
[14] https://huggingface.co/datasets/japanese-asr/whisper_transcriptions.mls
[15] https://huggingface.co/datasets/Akjava/QWEN3-TTS-Voice-Clone-100-Japanese-Female-ITA-Corpus-Emotion
[16] https://huggingface.co/datasets/tts-dataset/japanese-singing-voice