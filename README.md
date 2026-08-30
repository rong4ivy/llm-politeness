# Polite but Misaligned

Official repository for **“Polite but Misaligned: Evaluating LLM Politeness
Judgments Against Human Pragmatic Norms,”** accepted to Findings of EMNLP 2026.

This study evaluates how closely large language models (LLMs) align with human
politeness judgments. It considers two complementary evaluation settings:
continuous human ratings and three-way categorical labels (Impolite, Neutral,
and Polite). The analyses cover model--human and inter-model agreement,
prompt sensitivity, politeness strategies, directional label shifts, human
reference variation, and a diagnostic expert audit.

> **Repository status:** The repository is being prepared for the camera-ready
> release. Code and research artifacts listed below will be added before the
> archival release.

## Resources

The repository will include:

- model-evaluation and statistical-analysis code;
- the six prompt templates used in the study;
- data-preparation and sampling code;
- model outputs used in the reported analyses;
- annotations from the diagnostic expert audit;
- politeness-strategy detection and validation materials; and
- scripts for reproducing the paper's tables and figures.

## Models

The study evaluates five models through the OpenRouter API:

- Gemini 2.5 Flash
- GPT-4.1
- Claude 3.5 Sonnet
- Grok 3
- DeepSeek-V3

For the three-way categorical task, it additionally evaluates
Meta-Llama-3-8B-Instruct and Meta-Llama-3.1-8B-Instruct. The comparison between
the two Llama checkpoints is descriptive and is not treated as a controlled
test of model scale or post-training.

## Data

The evaluation uses two English-language datasets:

1. **Stanford Politeness Corpus** -- requests from Wikipedia Talk pages and
   Stack Exchange with continuous human politeness ratings.
2. **Politeness Corpus on Hugging Face** -- texts with three-way categorical
   labels, available from
   [`frfede/politeness-corpus`](https://huggingface.co/datasets/frfede/politeness-corpus).

The experiments use sampled evaluation sets rather than every item in the
source corpora. To respect the terms of the source datasets, the final release
will provide either the permitted evaluation files or the sampling information
needed to reconstruct them.

## Planned Repository Structure

```text
.
├── prompts/          # Prompt templates used in the evaluation
├── data/             # Data instructions, sampling code, and permitted files
├── outputs/          # Model outputs used in the analyses
├── annotations/      # Diagnostic expert-audit annotations
├── scripts/          # Evaluation and statistical-analysis scripts
├── figures/          # Figure-generation scripts and generated figures
└── README.md
```

The final file structure and exact reproduction commands will be documented
when the code and artifacts are uploaded.

## Reproducibility

The proprietary and open-weight models were queried through OpenRouter. No
temperature, top-p, or random seed was explicitly specified, and provider
routing was not constrained. The release will document model identifiers,
request handling, output parsing, and the commands used to reproduce the main
analyses.

The expert audit contains 318 deliberately selected high-disagreement and
strategy-conflict cases. It is intended as a diagnostic analysis and should not
be used to estimate the prevalence of errors in the full dataset.

## Citation

```bibtex
@inproceedings{wang-etal-2026-polite,
  title     = {Polite but Misaligned: Evaluating {LLM} Politeness Judgments
               Against Human Pragmatic Norms},
  author    = {Wang, Rong and Sun, Kun and Guo, Yadong},
  booktitle = {Findings of the Association for Computational Linguistics:
               {EMNLP} 2026},
  year      = {2026}
}
```

## License

A repository license will be added with the archival release. Third-party
datasets remain subject to their original licenses and terms of use.

## Contact

For questions about the paper or repository, please open an issue in this
repository.
