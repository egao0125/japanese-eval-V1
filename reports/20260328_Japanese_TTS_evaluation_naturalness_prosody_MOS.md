# Japanese TTS Evaluation: Naturalness, Prosody, MOS Scoring for Business Telephony

## Executive Summary

This report analyzes current methodologies for evaluating Japanese text-to-speech (TTS) systems in business telephony applications, focusing on naturalness, prosody, and Mean Opinion Score (MOS) assessment. While significant advances have been made in neural TTS architectures and evaluation frameworks, substantial gaps remain in telephony-specific evaluation metrics and standardized benchmarks for real-time Japanese TTS deployment in compressed audio environments.

## Key Findings

### Evaluation Metrics for Naturalness and Prosody

**Current State**: The research reveals limited telephony-specific evaluation frameworks for Japanese TTS. ESPnet2-TTS [1] demonstrates state-of-the-art performance on Japanese corpora with synthesized utterances reaching near ground-truth quality, but lacks telephony-specific evaluation metrics. The Speech BERT embedding approach [4] shows promise for capturing fine-grained prosodic patterns crucial for Japanese pitch accent systems, achieving improved objective quality metrics and subjective preference scores.

**Gap Identified**: No studies specifically address evaluation metrics tailored for Japanese TTS in telephony environments with compressed audio constraints.

### MOS Correlation with Objective Metrics

**Limited Findings**: While the French SSML prosody control study [5] demonstrates MOS improvements from 3.20 to 3.87 (p < 0.005) through automated prosody enhancement, no equivalent studies exist for Japanese telephony applications. BERTScore evaluation methodology [9] offers semantic similarity assessment that could complement traditional metrics, showing better correlation with human expert assessments than Word Error Rate alone.

**Research Gap**: No published research directly correlates MOS scores with pitch contour accuracy and phoneme duration specifically for Japanese TTS in business call scenarios.

### Critical Linguistic and Acoustic Features

**Prosodic Considerations**: Daisy-TTS [2] introduces prosody embedding decomposition for emotional expressiveness, which could address Japanese prosodic variation requirements. The zero-shot TTS approach [6] using self-supervised learning embeddings demonstrates improved rhythm transfer capabilities, potentially valuable for preserving Japanese prosodic patterns.

**Telephony-Specific Features**: No research specifically addresses acoustic feature optimization for Japanese TTS in compressed telephony environments.

### TTS Architecture Performance

**Speaker Adaptation**: The zero-shot TTS system [6] shows improved speaker similarity for unseen speakers without requiring speaker-specific training data. ESPnet2-TTS [1] provides extensive pre-trained Japanese models with unified recipe design for rapid prototyping.

**Emotional Expressiveness**: Daisy-TTS [2] demonstrates superior emotional speech naturalness through prosody embedding decomposition, enabling primary emotions, secondary emotions, and varying intensities - particularly relevant for Japanese communication nuances.

### Real-time Deployment Limitations

**Current Benchmarks**: Available Japanese TTS models show varying computational efficiency, with distilled Whisper models (japanese-asr/distil-whisper-large-v3-ja-reazonspeech series) offering efficiency-performance balance for ASR applications.

**Commercial Deployment Gaps**: No comprehensive benchmarks exist for real-time Japanese TTS deployment in commercial telephony systems, representing a significant research gap.

## Notable Papers

1. Hayashi, T., Yamamoto, R., Yoshimura, T., et al. (2021). "ESPnet2-TTS: Extending the Edge of TTS Research." *arXiv preprint arXiv:2110.07840v1*.

2. Chevi, R., & Aji, A. F. (2024). "Daisy-TTS: Simulating Wider Spectrum of Emotions via Prosody Embedding Decomposition." *arXiv preprint arXiv:2402.14523v2*.

3. Chen, L., Deng, Y., Wang, X., et al. (2021). "Speech BERT Embedding For Improving Prosody in Neural TTS." *arXiv preprint arXiv:2106.04312v3*.

4. Ould Ouali, N., Sani, A. H., Bueno, R., et al. (2025). "Improving French Synthetic Speech Quality via SSML Prosody Control." *arXiv preprint arXiv:2508.17494v1*.

5. Fujita, K., Ashihara, T., Kanagawa, H., et al. (2023). "Zero-shot text-to-speech synthesis conditioned using self-supervised speech representation model." *arXiv preprint arXiv:2304.11976v1*.

## Notable Tools & Models

### GitHub Repositories
- **myshell-ai/MeloTTS** (7,299 stars) - Multi-lingual TTS library with Japanese support
- **reazon-research/ReazonSpeech** (376 stars) - Massive open Japanese speech corpus
- **atomicoo/FCH-TTS** (281 stars) - Fast multilingual TTS model supporting Japanese

### HuggingFace Models
- **japanese-asr/distil-whisper-large-v3-ja-reazonspeech-all** (24 downloads) - Japanese-optimized distilled Whisper ASR
- **2121-8/japanese-parler-tts-mini** (1,082 downloads) - Japanese TTS model
- **esnya/japanese_speecht5_tts** (374 downloads) - Japanese SpeechT5 TTS implementation
- **Bagus/wav2vec2-xlsr-japanese-speech-emotion-recognition** (410 downloads) - Japanese emotion recognition

## Available Datasets

### High-Volume Datasets
- **japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized** (9,065 downloads)
- **japanese-asr/whisper_transcriptions.mls** (3,397 downloads)
- **Akjava/QWEN3-TTS-Voice-Clone-100-Japanese-Female-ITA-Corpus-Emotion** (7,140 downloads)

### Specialized TTS Datasets
- **tts-dataset/japanese-singing-voice** (265 downloads)
- **Akjava/QWEN3-TTS-Voice-Design-100-Japanese-Female-Designed-Voices** (749 downloads)

## Recommendations for Japanese Voice AI Evaluation

### Immediate Actions
1. **Develop Telephony-Specific Evaluation Framework**: Establish standardized metrics for Japanese TTS quality assessment in compressed audio environments, incorporating pitch contour accuracy and phoneme duration measurements.

2. **Create Japanese Business Telephony Corpus**: Build specialized datasets reflecting business communication scenarios with appropriate emotional and prosodic variations.

3. **Implement Multi-Modal Evaluation**: Adopt BERTScore-based semantic evaluation alongside traditional MOS scoring for more comprehensive quality assessment.

### Medium-Term Development
1. **Establish Real-Time Performance Benchmarks**: Define computational and quality benchmarks for commercial Japanese TTS deployment in telephony systems.

2. **Prosody-Aware Evaluation Metrics**: Develop evaluation methodologies that capture Japanese-specific prosodic features including pitch accent patterns and emotional expressiveness.

3. **Cross-Architecture Comparison Studies**: Conduct systematic evaluations comparing ESPnet2-TTS, Daisy-TTS, and other architectures on standardized Japanese telephony tasks.

### Long-Term Research Directions
1. **Cultural Context Integration**: Incorporate culturally-aware evaluation methods following JMMMU benchmark principles for business communication appropriateness.

2. **Adaptive Quality Assessment**: Develop dynamic evaluation systems that adjust quality metrics based on telephony compression levels and network conditions.

## References

1. Tomoki Hayashi, Ryuichi Yamamoto, Takenori Yoshimura et al. "ESPnet2-TTS: Extending the Edge of TTS Research" (2021). http://arxiv.org/abs/2110.07840v1

2. Rendi Chevi, Alham Fikri Aji. "Daisy-TTS: Simulating Wider Spectrum of Emotions via Prosody Embedding Decomposition" (2024). http://arxiv.org/abs/2402.14523v2

3. Shota Onohara, Atsuyuki Miyai, Yuki Imajuku et al. "JMMMU: A Japanese Massive Multi-discipline Multimodal Understanding Benchmark for Culture-aware Evaluation" (2024). http://arxiv.org/abs/2410.17250v2

4. Liping Chen, Yan Deng, Xi Wang et al. "Speech BERT Embedding For Improving Prosody in Neural TTS" (2021). http://arxiv.org/abs/2106.04312v3

5. Nassima Ould Ouali, Awais Hussain Sani, Ruben Bueno et al. "Improving French Synthetic Speech Quality via SSML Prosody Control" (2025). http://arxiv.org/abs/2508.17494v1

6. Kenichi Fujita, Takanori Ashihara, Hiroki Kanagawa et al. "Zero-shot text-to-speech synthesis conditioned using self-supervised speech representation model" (2023). http://arxiv.org/abs/2304.11976v1

7. Jun Fu. "Scale Guided Hypernetwork for Blind Super-Resolution Image Quality Assessment" (2023). http://arxiv.org/abs/2306.02398v1

8. Wei Sun, Linhan Cao, Jun Jia et al. "Enhancing Blind Video Quality Assessment with Rich Quality-aware Features" (2024). http://arxiv.org/abs/2405.08745v2

9. Jimmy Tobin, Qisheng Li, Subhashini Venugopalan et al. "Assessing ASR Model Quality on Disordered Speech using BERTScore" (2022). http://arxiv.org/abs/2209.10591v1

10. Abir Elmir, Badr Elmir, Bouchaib Bounabat. "Towards an Assessment-oriented Model for External Information System Quality Characterization" (2013). http://arxiv.org/abs/1310.8111v1