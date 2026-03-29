# LLM-as-Judge Evaluation Methods for Voice Agents: Automated Quality Assessment, Conversation Scoring, and Multi-Dimensional Rubrics for Japanese Business Calls

## Executive Summary

Current research in LLM-based speech evaluation provides foundational methodologies through prompting frameworks like SpeechPrompt and curriculum learning approaches, but significant gaps remain in developing comprehensive evaluation systems specifically for Japanese voice agents in business contexts. The available literature demonstrates progress in language-agnostic speech processing and multi-modal evaluation techniques, yet lacks dedicated frameworks for handling Japanese honorific language patterns, cultural appropriateness assessment, and code-switching scenarios in telephony applications.

## Key Findings

### How can LLMs effectively evaluate Japanese voice agent performance in business call scenarios with domain-specific terminology and honorific language patterns?

**Research Gap Identified**: The current literature does not directly address Japanese-specific evaluation methodologies. However, relevant foundational work includes:

- **SpeechPrompt framework** [1] demonstrates that speech-to-unit conversion enables LLMs to process speech tasks with minimal parameter updates, suggesting potential for adapting text-based Japanese language models to speech evaluation
- **Language-agnostic approaches** from CUPE [4] show that phoneme-level processing can maintain cross-lingual effectiveness, which could be adapted for Japanese phoneme recognition in business contexts
- **Multi-modal integration** [5] suggests that combining audio and visual information improves speech processing accuracy, potentially applicable to video call scenarios in Japanese business settings

**Gap**: No existing research specifically addresses honorific language pattern evaluation or domain-specific terminology assessment for Japanese voice agents.

### What multi-dimensional rubrics are most suitable for assessing Japanese business call quality including politeness, accuracy, and cultural appropriateness?

**Research Gap Identified**: Current literature lacks comprehensive multi-dimensional rubrics for Japanese business communication evaluation.

- **Curriculum learning approaches** [2] demonstrate that inter-annotator disagreement can serve as a quality signal, suggesting that cultural nuances in Japanese communication could be quantified through annotator consensus metrics
- **Semantic retrieval methods** [3] show that models can learn to identify semantic matches that align with human judgments, indicating potential for developing cultural appropriateness metrics

**Gap**: No established rubrics exist for evaluating Japanese business call quality across politeness, accuracy, and cultural dimensions.

### How do automated LLM-based scoring methods compare to human evaluation for Japanese voice agents in terms of correlation and reliability?

**Research Gap Identified**: No comparative studies exist between automated LLM scoring and human evaluation specifically for Japanese voice agents.

- **Human agreement metrics** [2] provide methodological foundations for measuring reliability through inter-annotator agreement
- **Semantic alignment studies** [3] show that automated systems can achieve ~60% precision in semantic speech retrieval tasks and often outperform supervised models in finding matches that align with human judgments

**Gap**: Correlation and reliability studies between LLM-based and human evaluation for Japanese speech are not available in current literature.

### What are the optimal prompt engineering strategies for LLM judges to evaluate Japanese speech-to-speech conversation flow and naturalness?

**Research Gap Identified**: Specific prompt engineering strategies for Japanese speech evaluation are not addressed in current research.

- **SpeechPrompt methodology** [1] demonstrates that reformulating speech tasks as speech-to-unit generation problems enables unified processing, suggesting potential prompt structures for conversation evaluation
- **Contextless processing** [4] shows that phoneme-level analysis without context can maintain effectiveness, indicating that naturalness evaluation might benefit from local rather than global prompting strategies

**Gap**: No optimal prompting strategies have been established for Japanese speech-to-speech evaluation scenarios.

### How can LLM evaluation frameworks handle code-switching between Japanese and English in business telephony scenarios?

**Research Gap Identified**: Code-switching evaluation frameworks are not addressed in the available literature.

- **Language-agnostic processing** [4] provides foundations through universal phoneme encoding that works across languages
- **Cross-lingual capabilities** demonstrated in multiple papers suggest technical feasibility for handling language switching

**Gap**: No frameworks exist specifically for evaluating code-switching in Japanese-English business telephony contexts.

## Notable Papers

1. Chang, K.-W., Wu, H., Wang, Y.-K., et al. (2024). SpeechPrompt: Prompting Speech Language Models for Speech Processing Tasks. *arXiv preprint arXiv:2408.13040v1*. Categories: eess.AS, cs.AI, cs.CL, cs.LG.

2. Lotfian, R., Busso, C. (2018). Curriculum Learning for Speech Emotion Recognition from Crowdsourced Labels. *arXiv preprint arXiv:1805.10339v1*. Categories: eess.AS, cs.SD.

3. Kamper, H., Shakhnarovich, G., Livescu, K. (2017). Semantic speech retrieval with a visually grounded model of untranscribed speech. *arXiv preprint arXiv:1710.01949v2*. Categories: cs.CL, cs.CV, eess.AS.

4. Rehman, A., Zhang, J.-J., Yang, X. (2025). CUPE: Contextless Universal Phoneme Encoder for Language-Agnostic Speech Processing. *arXiv preprint arXiv:2508.15316v1*. Categories: cs.CL, cs.LG, eess.AS.

5. Sadeghi, M., Leglaive, S., Alameda-Pineda, X., et al. (2019). Audio-visual Speech Enhancement Using Conditional Variational Auto-Encoders. *arXiv preprint arXiv:1908.02590v3*. Categories: cs.SD, cs.LG, eess.AS.

## Notable Tools & Models

**Information Gap**: The provided research papers do not include specific tool repositories, GitHub links, or download statistics. The papers represent theoretical and methodological contributions without publicly available implementation details.

## Available Datasets

**Information Gap**: The research findings do not provide comprehensive dataset information. Mentioned datasets include:
- UCLA Phonetic Corpus [4]
- NTCD-TIMIT dataset [5]
- GRID dataset [5]

No Japanese-specific business call datasets are identified in the current literature.

## Recommendations for Japanese Voice AI Evaluation

### Immediate Development Priorities

1. **Establish Japanese-specific evaluation frameworks** by adapting SpeechPrompt methodology [1] to handle Japanese phoneme sequences and honorific patterns
2. **Develop multi-dimensional rubrics** incorporating curriculum learning principles [2] to handle cultural nuances in Japanese business communication
3. **Create benchmark datasets** for Japanese business calls including code-switching scenarios and cultural appropriateness annotations

### Technical Implementation Strategies

1. **Leverage language-agnostic phoneme processing** [4] as a foundation while developing Japanese-specific semantic layers
2. **Implement multi-modal evaluation** [5] for video call scenarios common in Japanese business contexts
3. **Design prompt engineering frameworks** based on speech-to-unit conversion principles [1] for LLM-based evaluation

### Research Gaps Requiring Investigation

1. **Correlation studies** between automated LLM scoring and human evaluation for Japanese speech
2. **Code-switching evaluation frameworks** for Japanese-English business telephony
3. **Cultural appropriateness metrics** specific to Japanese business communication norms
4. **Honorific language pattern recognition** and evaluation methodologies

## References

1. Chang, K.-W., Wu, H., Wang, Y.-K., et al. (2024). SpeechPrompt: Prompting Speech Language Models for Speech Processing Tasks. http://arxiv.org/abs/2408.13040v1

2. Lotfian, R., Busso, C. (2018). Curriculum Learning for Speech Emotion Recognition from Crowdsourced Labels. http://arxiv.org/abs/1805.10339v1

3. Kamper, H., Shakhnarovich, G., Livescu, K. (2017). Semantic speech retrieval with a visually grounded model of untranscribed speech. http://arxiv.org/abs/1710.01949v2

4. Rehman, A., Zhang, J.-J., Yang, X. (2025). CUPE: Contextless Universal Phoneme Encoder for Language-Agnostic Speech Processing. http://arxiv.org/abs/2508.15316v1

5. Sadeghi, M., Leglaive, S., Alameda-Pineda, X., et al. (2019). Audio-visual Speech Enhancement Using Conditional Variational Auto-Encoders. http://arxiv.org/abs/1908.02590v3