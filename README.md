# SarcasmDetector

AI-Assisted Sarcasm Detection in Reddit Conversations — EECS 543 Assignment 2

## Overview

This project implements a human-AI collaborative task for detecting sarcasm in Reddit conversations. The AI model (Claude Haiku 4.5) provides predictions with confidence scores and reasoning explanations that human participants can use to make better decisions.

## Dataset

- **Source**: [FigLang 2020 Sarcasm Detection Shared Task](https://github.com/EducationalTestingService/sarcasm)
- **Study Dataset**: 250 curated trials (125 sarcastic, 125 not sarcastic)
- **Format**: Each trial includes thread context (2-4 comments), target response, and ground truth label

## Files

| File | Description |
|------|-------------|
| `A2_Report.pdf` | Full assignment report with task description, evaluation, and predictions |
| `study_dataset.json` | Curated study dataset (250 trials) |
| `study_dataset_final.csv` | Study dataset with model outputs (CSV format) |
| `model_outputs.json` | Complete model predictions with reasoning |
| `run_inference.py` | Script to run Claude Haiku on the dataset |
| `evaluate_model.py` | Evaluation and error analysis script |

## Replicating Results

### Run Inference
```bash
pip install anthropic
export ANTHROPIC_API_KEY="your-key-here"
python run_inference.py
```

### Run Evaluation
```bash
python evaluate_model.py
```

## Results

| Metric | Value |
|--------|-------|
| Accuracy | 69.6% |
| Precision (SARCASM) | 64.5% |
| Recall (SARCASM) | 87.2% |
| F1 Score | 0.741 |

**Predicted Ranking**: Human-AI Joint > Human-Only > AI-Only
