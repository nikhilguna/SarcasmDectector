# Sarcasm Detector

AI-Assisted Sarcasm Detection in Reddit Conversations — EECS 543 Assignments 2 & 3

## Overview

This project implements a human-AI collaborative task for detecting sarcasm in Reddit conversations. It includes:

1. **Assignment 2**: AI model evaluation using Claude Haiku 4.5 for sarcasm classification
2. **Assignment 3**: Human-subjects study comparing performance with and without AI assistance

### Key Results

| Condition | Accuracy | Notes |
|-----------|----------|-------|
| AI-Only (Claude Haiku) | 69.6% | From Assignment 2 |
| Human + AI | 66.7% | With AI assistance showing prediction, confidence, reasoning |
| Human Only | 57.0% | Baseline, no AI assistance |

**A2 Prediction**: Human-AI Joint > Human-Only > AI-Only
**Actual Result**: AI-Only > Human+AI > Human-Only

While AI assistance improved human accuracy by ~10%, humans did not outperform the AI alone. Analysis shows humans over-trusted the AI (88.6% agreement rate) but when they did override, they were often wrong (31.2% success rate).

## Repository Structure

```
SarcasmDetector/
├── README.md                    # This file
├── .env.example                 # Environment variable template
│
├── # Assignment 2 - AI Model Evaluation
├── run_inference.py             # Run Claude Haiku on the dataset
├── evaluate_model.py            # Compute metrics and error analysis
├── model_outputs.json           # Complete AI predictions (250 trials)
├── study_dataset.json           # Curated study dataset
├── study_dataset_final.csv      # Dataset with model outputs (CSV)
│
├── # Assignment 3 - Human Study Analysis
├── analyze_results.py           # Main analysis script for A3
├── data/
│   ├── sessions.csv             # Participant session data
│   ├── responses.csv            # Individual trial responses
│   ├── model_outputs.json       # AI predictions (copy for analysis)
│   └── analysis_output/         # Generated charts and tables
│       ├── accuracy_by_condition.png
│       ├── participant_accuracy_dotplot.png
│       ├── override_analysis_pie.png
│       ├── response_time_comparison.png
│       ├── participant_accuracy_all.csv
│       ├── participant_accuracy_clean.csv
│       ├── condition_summary_all.csv
│       └── condition_summary_clean.csv
│
├── # Web Application (Study Interface)
├── src/                         # Next.js app source
│   ├── app/
│   │   ├── page.tsx            # Landing page
│   │   ├── no-ai/              # Human-only condition
│   │   ├── with-ai/            # Human+AI condition
│   │   ├── admin/              # Results dashboard
│   │   └── api/                # API routes
│   └── lib/                    # Utilities
├── package.json                 # Node.js dependencies
├── next.config.js              # Next.js configuration
├── tailwind.config.ts          # Tailwind CSS configuration
└── tsconfig.json               # TypeScript configuration
```

## Dataset

- **Source**: [FigLang 2020 Sarcasm Detection Shared Task](https://github.com/EducationalTestingService/sarcasm)
- **Study Dataset**: 250 curated trials (125 sarcastic, 125 not sarcastic)
- **Format**: Each trial includes thread context (2-4 comments), target response, and ground truth label

## Setup & Usage

### Assignment 2: Run AI Model Inference

```bash
# Install dependencies
pip install anthropic

# Set API key
export ANTHROPIC_API_KEY="your-key-here"

# Run inference on the dataset
python run_inference.py

# Evaluate model performance
python evaluate_model.py
```

**Output metrics** (from evaluate_model.py):
- Accuracy: 69.6%
- Precision: 64.5%
- Recall: 87.2%
- F1 Score: 0.741

### Assignment 3: Run Study Analysis

```bash
# Create virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install pandas numpy scipy matplotlib

# Run analysis
python analyze_results.py
```

The script will:
1. Load and clean data (remove duplicates, flag outliers)
2. Calculate per-participant and per-condition accuracy
3. Run Mann-Whitney U statistical test
4. Analyze response times
5. Compute override patterns (human agreement/disagreement with AI)
6. Generate charts (saved to `data/analysis_output/`)
7. Print summary for report

### Run the Web Application Locally

```bash
# Install Node.js dependencies
npm install

# Set up environment variables
cp .env.example .env.local
# Edit .env.local with your Supabase credentials

# Start development server
npm run dev
```

Open http://localhost:3000 in your browser.

## Web Application

The study interface is built with Next.js and deployed on Vercel, with Supabase as the backend database.

### URLs

- **Production**: https://sarcasm-study.vercel.app
  - No-AI condition: `/no-ai`
  - With-AI condition: `/with-ai`
  - Admin dashboard: `/admin`

### Conditions

**No-AI (Baseline)**:
- Shows Reddit thread context and target response
- Participant classifies as SARCASM or NOT_SARCASM
- No assistance provided

**With-AI**:
- Same task as baseline
- Additionally shows Claude Haiku's prediction:
  - Classification (SARCASM / NOT_SARCASM)
  - Confidence score (0-100%)
  - Reasoning explanation
- Participant makes final decision (can agree or override AI)

### Database Schema

**sessions** table:
- `worker_id`: Unique participant identifier
- `condition`: 'no-ai' or 'with-ai'
- `completion_code`: Code for MTurk verification
- `trials_completed`: Number of trials (target: 20)
- `accuracy`: Calculated accuracy
- `created_at`: Timestamp

**responses** table:
- `worker_id`: Links to sessions
- `condition`: Condition type
- `trial_id`: Trial number (1-250)
- `ground_truth`: Correct answer
- `participant_answer`: Participant's response
- `time_spent_ms`: Response time in milliseconds
- `created_at`: Timestamp

### Admin Dashboard

The `/admin` route displays:
- Total participants and per-condition counts
- Aggregate accuracy by condition
- Per-participant breakdown table

Requires `SUPABASE_SERVICE_ROLE_KEY` in environment variables.

## Analysis Details

### Data Cleaning

The analysis script handles:
- Duplicate sessions (same worker, multiple attempts)
- Duplicate responses (same worker + trial)
- Flagged outliers:
  - P-MN4SB1JM-XI7KWK: 35% accuracy (likely inverted labels)
  - P-MN4QRFUY-5YFWZO: Multiple trials under 2 seconds (low effort)

Results are reported both with and without outliers.

### Statistical Tests

- **Mann-Whitney U test**: Non-parametric comparison of accuracy distributions
- **Permutation test**: Robustness check (10,000 permutations)
- Both tests run with all data and excluding outliers

### Override Analysis

For with-AI trials, each response is categorized:
- **Both Correct**: AI correct AND human correct (agreement, both right)
- **Good Override**: AI wrong BUT human correct (human corrected AI)
- **Bad Override**: AI correct BUT human wrong (human overrode correct AI)
- **Both Wrong**: AI wrong AND human wrong

Key metrics:
- Agreement rate: How often human matched AI prediction
- Override rate: How often human disagreed with AI
- Override success rate: When human overrode, how often were they correct

## MTurk Integration

The study was deployed via Amazon Mechanical Turk Sandbox:
1. HITs posted as survey links
2. Participants complete 20 trials through the web interface
3. Completion code displayed at end
4. Participants submit code to MTurk for approval

## Results Summary

### Accuracy

| Condition | Mean Accuracy | Std Dev | N |
|-----------|--------------|---------|---|
| No-AI | 57.0% | 5.7% | 5 |
| With-AI | 66.7% | 10.8% | 6 |
| AI-Only | 69.6% | - | - |

### Statistical Significance

- Mann-Whitney U: p = 0.071 (not significant at p < 0.05)
- Effect size (rank-biserial r): 0.62 (large effect)
- Small sample size limits statistical power

### Override Patterns

- Agreement rate: 88.6%
- Override rate: 11.4%
- Override success: 31.2%

Humans strongly follow AI recommendations but are often wrong when they disagree.

## Key Findings

1. **AI assistance helps**: Human+AI (66.7%) > Human-only (57.0%)
2. **AI alone is best**: AI-only (69.6%) > Human+AI (66.7%)
3. **Over-reliance**: 88.6% agreement rate suggests humans defer to AI too much
4. **Poor calibration**: When humans override AI, they're wrong 68.8% of the time
5. **Error patterns**: Humans miss more sarcasm (38.6% FN) than AI (12.8% FN)

## License

This project was created for EECS 543 at the University of Michigan. Dataset sourced from the FigLang 2020 Shared Task.

## Author

Nikhil Guna
EECS 543 - AI Ethics
University of Michigan, Winter 2026
