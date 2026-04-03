#!/usr/bin/env python3
"""
Human-AI Sarcasm Detection Study Analysis
==========================================
EECS 543 Assignment 3-2

This script analyzes the results of a human-subjects study comparing sarcasm
detection performance with and without AI assistance.

Study Design:
- Task: Binary classification of Reddit comments as SARCASM or NOT_SARCASM
- Two conditions: no-ai (baseline) and with-ai (shows Claude Haiku prediction)
- AI-Only accuracy from Assignment 2: 69.6% (174/250)
- Prediction: Human-AI Joint > Human-Only > AI-Only

Data Files (in data/ directory):
- sessions.csv: One row per participant session
- responses.csv: One row per trial response
- model_outputs.json: AI model predictions from Assignment 2

Usage:
    python analyze_results.py

Output:
- Prints analysis results to stdout
- Saves summary tables as CSVs in data/analysis_output/
- Saves charts as PNGs in data/analysis_output/

Requirements:
    pip install pandas numpy scipy matplotlib

Author: Nikhil Guna
Date: April 2026
"""

import json
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

# Constants
AI_ONLY_ACCURACY = 0.696  # From Assignment 2: 174/250
AI_ONLY_PRECISION = 0.645
AI_ONLY_RECALL = 0.872
AI_ONLY_F1 = 0.741

# Outliers to flag
OUTLIER_LOW_ACCURACY = 'P-MN4SB1JM-XI7KWK'  # 35% accuracy in no-ai
OUTLIER_LOW_EFFORT = 'P-MN4QRFUY-5YFWZO'    # Very fast responses in with-ai

# Data paths
DATA_DIR = Path('data')
OUTPUT_DIR = DATA_DIR / 'analysis_output'


def load_data():
    """Load all data files."""
    sessions = pd.read_csv(DATA_DIR / 'sessions.csv')
    responses = pd.read_csv(DATA_DIR / 'responses.csv')

    with open(DATA_DIR / 'model_outputs.json') as f:
        model_outputs = json.load(f)

    # Create lookup dict for model predictions
    model_predictions = {
        item['trial_id']: {
            'model_prediction': item['model_prediction'],
            'ground_truth': item['ground_truth'],
            'model_confidence': item['model_confidence']
        }
        for item in model_outputs
    }

    return sessions, responses, model_predictions


def clean_data(sessions, responses):
    """Clean data by removing duplicates and flagging outliers."""
    print("=" * 70)
    print("STEP 1: DATA LOADING AND CLEANING")
    print("=" * 70)

    print(f"\nRaw data loaded:")
    print(f"  Sessions: {len(sessions)} rows")
    print(f"  Responses: {len(responses)} rows")

    # --- Session Cleaning ---
    print("\n--- Session Cleaning ---")

    # Identify duplicate sessions
    duplicate_sessions = sessions[sessions.duplicated(subset=['worker_id', 'condition'], keep=False)]
    print(f"\nDuplicate sessions found:")
    for worker_id in duplicate_sessions['worker_id'].unique():
        worker_sessions = sessions[sessions['worker_id'] == worker_id]
        print(f"  {worker_id}:")
        for _, row in worker_sessions.iterrows():
            print(f"    - {row['completion_code']}: {row['trials_completed']} trials, "
                  f"accuracy={row['accuracy']:.3f}, created={row['created_at']}")

    # Remove duplicate session for P-MN4QFLHS-WO5HMW (keep first, 20 trials)
    # Also remove duplicate session for P-MNJ0SBB7-Y2S0PH (keep first, 20 trials)
    sessions_clean = sessions.copy()

    # Keep only first session per worker_id + condition combination
    sessions_clean = sessions_clean.sort_values('created_at')
    sessions_clean = sessions_clean.drop_duplicates(subset=['worker_id', 'condition'], keep='first')

    print(f"\nAfter removing duplicate sessions: {len(sessions_clean)} sessions")

    # Remove incomplete sessions (participants who appear in sessions but didn't complete properly)
    # Filter to only workers who completed exactly 20 trials
    sessions_clean = sessions_clean[sessions_clean['trials_completed'] == 20]
    print(f"After filtering to 20-trial sessions: {len(sessions_clean)} sessions")

    # --- Response Cleaning ---
    print("\n--- Response Cleaning ---")

    # Find valid worker_ids from cleaned sessions
    valid_workers = set(sessions_clean['worker_id'])

    # Filter responses to only include valid workers
    responses_clean = responses[responses['worker_id'].isin(valid_workers)].copy()
    print(f"After filtering to valid workers: {len(responses_clean)} responses")

    # Remove duplicate responses (same worker, same trial)
    dup_responses = responses_clean[responses_clean.duplicated(subset=['worker_id', 'trial_id'], keep=False)]
    if len(dup_responses) > 0:
        print(f"\nDuplicate responses found:")
        for _, row in dup_responses.iterrows():
            print(f"  Worker {row['worker_id']}, Trial {row['trial_id']}, Time {row['time_spent_ms']}ms")

    responses_clean = responses_clean.sort_values('created_at')
    responses_clean = responses_clean.drop_duplicates(subset=['worker_id', 'trial_id'], keep='first')
    print(f"After removing duplicate responses: {len(responses_clean)} responses")

    # --- Outlier Identification ---
    print("\n--- Outlier Identification ---")

    # Flag low accuracy outlier
    low_acc = sessions_clean[sessions_clean['worker_id'] == OUTLIER_LOW_ACCURACY]
    if len(low_acc) > 0:
        acc = low_acc.iloc[0]['accuracy']
        print(f"\n[OUTLIER] {OUTLIER_LOW_ACCURACY}:")
        print(f"  Accuracy: {acc:.1%} (suspiciously low)")
        print(f"  Condition: {low_acc.iloc[0]['condition']}")
        print(f"  Interpretation: Likely misunderstood task or inverted labels")

    # Flag low effort outlier (check response times)
    low_effort_responses = responses_clean[responses_clean['worker_id'] == OUTLIER_LOW_EFFORT]
    if len(low_effort_responses) > 0:
        fast_trials = low_effort_responses[low_effort_responses['time_spent_ms'] < 2000]
        very_fast = low_effort_responses[low_effort_responses['time_spent_ms'] < 1000]
        print(f"\n[OUTLIER] {OUTLIER_LOW_EFFORT}:")
        print(f"  Trials under 2 seconds: {len(fast_trials)}/{len(low_effort_responses)}")
        print(f"  Trials under 1 second: {len(very_fast)}/{len(low_effort_responses)}")
        print(f"  Condition: {low_effort_responses.iloc[0]['condition']}")
        print(f"  Interpretation: Clicking through without reading")

    # --- Summary ---
    print("\n--- Cleaning Summary ---")
    print(f"Sessions: {len(sessions)} -> {len(sessions_clean)} (removed {len(sessions) - len(sessions_clean)})")
    print(f"Responses: {len(responses)} -> {len(responses_clean)} (removed {len(responses) - len(responses_clean)})")

    return sessions_clean, responses_clean


def compute_participant_accuracy(responses):
    """Compute accuracy for each participant."""
    participant_stats = responses.groupby(['worker_id', 'condition']).apply(
        lambda g: pd.Series({
            'num_correct': (g['ground_truth'] == g['participant_answer']).sum(),
            'num_trials': len(g),
            'accuracy': (g['ground_truth'] == g['participant_answer']).mean()
        })
    ).reset_index()
    return participant_stats


def compute_condition_summary(participant_stats):
    """Compute summary statistics per condition."""
    summary = participant_stats.groupby('condition').agg({
        'accuracy': ['count', 'mean', 'std', 'median', 'min', 'max']
    }).reset_index()
    summary.columns = ['condition', 'num_participants', 'mean_accuracy', 'std_dev',
                       'median', 'min', 'max']
    return summary


def performance_comparison(responses, outliers_to_exclude=None):
    """Compare performance across conditions."""
    df = responses.copy()

    # Exclude outliers if specified
    if outliers_to_exclude:
        df = df[~df['worker_id'].isin(outliers_to_exclude)]

    participant_stats = compute_participant_accuracy(df)
    condition_summary = compute_condition_summary(participant_stats)

    return participant_stats, condition_summary


def print_performance_results(participant_stats, condition_summary, label=""):
    """Print formatted performance results."""
    print(f"\n--- Per-Participant Accuracy {label} ---")
    print(participant_stats.to_string(index=False))

    print(f"\n--- Per-Condition Summary {label} ---")
    print(condition_summary.to_string(index=False))

    # Add AI-Only comparison
    print(f"\nComparison with AI-Only baseline: {AI_ONLY_ACCURACY:.1%}")

    # Determine ranking
    no_ai_mean = condition_summary[condition_summary['condition'] == 'no-ai']['mean_accuracy'].values[0]
    with_ai_mean = condition_summary[condition_summary['condition'] == 'with-ai']['mean_accuracy'].values[0]

    rankings = [
        ('No-AI (Human Only)', no_ai_mean),
        ('With-AI (Human+AI)', with_ai_mean),
        ('AI-Only', AI_ONLY_ACCURACY)
    ]
    rankings.sort(key=lambda x: x[1], reverse=True)

    print(f"\nRanking {label}:")
    for i, (name, acc) in enumerate(rankings, 1):
        print(f"  {i}. {name}: {acc:.1%}")

    return rankings


def mann_whitney_test(participant_stats):
    """Run Mann-Whitney U test comparing conditions."""
    no_ai = participant_stats[participant_stats['condition'] == 'no-ai']['accuracy'].values
    with_ai = participant_stats[participant_stats['condition'] == 'with-ai']['accuracy'].values

    # Mann-Whitney U test
    statistic, p_value = stats.mannwhitneyu(no_ai, with_ai, alternative='two-sided')

    # Effect size: rank-biserial correlation
    n1, n2 = len(no_ai), len(with_ai)
    r = 1 - (2 * statistic) / (n1 * n2)

    return {
        'U': statistic,
        'p_value': p_value,
        'effect_size': r,
        'n1': n1,
        'n2': n2,
        'no_ai_mean': np.mean(no_ai),
        'with_ai_mean': np.mean(with_ai)
    }


def permutation_test(participant_stats, n_permutations=10000):
    """Run permutation test as robustness check."""
    no_ai = participant_stats[participant_stats['condition'] == 'no-ai']['accuracy'].values
    with_ai = participant_stats[participant_stats['condition'] == 'with-ai']['accuracy'].values

    observed_diff = np.mean(with_ai) - np.mean(no_ai)
    combined = np.concatenate([no_ai, with_ai])
    n1 = len(no_ai)

    count_extreme = 0
    np.random.seed(42)  # For reproducibility

    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_diff = np.mean(combined[n1:]) - np.mean(combined[:n1])
        if abs(perm_diff) >= abs(observed_diff):
            count_extreme += 1

    p_value = count_extreme / n_permutations
    return {
        'observed_diff': observed_diff,
        'p_value': p_value,
        'n_permutations': n_permutations
    }


def run_statistical_tests(responses, outliers=None):
    """Run all statistical tests."""
    print("\n" + "=" * 70)
    print("STEP 3: STATISTICAL TESTS")
    print("=" * 70)

    # Test with all data
    print("\n--- With All Participants ---")
    participant_stats = compute_participant_accuracy(responses)
    results_all = mann_whitney_test(participant_stats)
    perm_all = permutation_test(participant_stats)

    print(f"\nMann-Whitney U Test:")
    print(f"  U statistic: {results_all['U']:.1f}")
    print(f"  p-value: {results_all['p_value']:.4f}")
    print(f"  Effect size (rank-biserial r): {results_all['effect_size']:.3f}")
    print(f"  n(no-ai)={results_all['n1']}, n(with-ai)={results_all['n2']}")
    sig = "YES" if results_all['p_value'] < 0.05 else "NO"
    print(f"  Significant at p < 0.05: {sig}")

    print(f"\nPermutation Test (robustness check):")
    print(f"  Observed difference (with-ai - no-ai): {perm_all['observed_diff']:.3f}")
    print(f"  p-value ({perm_all['n_permutations']} permutations): {perm_all['p_value']:.4f}")

    # Test without outliers
    if outliers:
        print("\n--- Without Outliers ---")
        responses_clean = responses[~responses['worker_id'].isin(outliers)]
        participant_stats_clean = compute_participant_accuracy(responses_clean)
        results_clean = mann_whitney_test(participant_stats_clean)
        perm_clean = permutation_test(participant_stats_clean)

        print(f"\nMann-Whitney U Test:")
        print(f"  U statistic: {results_clean['U']:.1f}")
        print(f"  p-value: {results_clean['p_value']:.4f}")
        print(f"  Effect size (rank-biserial r): {results_clean['effect_size']:.3f}")
        print(f"  n(no-ai)={results_clean['n1']}, n(with-ai)={results_clean['n2']}")
        sig = "YES" if results_clean['p_value'] < 0.05 else "NO"
        print(f"  Significant at p < 0.05: {sig}")

        print(f"\nPermutation Test:")
        print(f"  Observed difference: {perm_clean['observed_diff']:.3f}")
        print(f"  p-value: {perm_clean['p_value']:.4f}")

    print("\n--- Limitations ---")
    print("  - Small sample size limits statistical power")
    print("  - Non-parametric test used due to small samples and potential non-normality")
    print("  - Results should be interpreted with caution")

    return results_all, perm_all


def time_analysis(responses):
    """Analyze response times."""
    print("\n" + "=" * 70)
    print("STEP 4: TIME ANALYSIS")
    print("=" * 70)

    # Filter out very fast responses (likely errors)
    df = responses[responses['time_spent_ms'] >= 1000].copy()
    print(f"\nFiltered out {len(responses) - len(df)} responses under 1 second")

    # Convert to seconds
    df['time_spent_sec'] = df['time_spent_ms'] / 1000

    # Per-condition statistics
    print("\n--- Response Time by Condition ---")
    time_stats = df.groupby('condition')['time_spent_sec'].agg([
        'mean', 'median', 'std', 'min', 'max', 'count'
    ]).round(2)
    time_stats.columns = ['Mean (s)', 'Median (s)', 'Std Dev', 'Min (s)', 'Max (s)', 'N']
    print(time_stats)

    # Per-participant average time
    print("\n--- Average Response Time per Participant ---")
    participant_time = df.groupby(['worker_id', 'condition'])['time_spent_sec'].mean().reset_index()
    participant_time.columns = ['worker_id', 'condition', 'avg_time_sec']
    participant_time = participant_time.round(2)
    print(participant_time.to_string(index=False))

    return time_stats, participant_time


def error_pattern_analysis(responses, model_predictions):
    """Analyze error patterns, especially for with-ai condition."""
    print("\n" + "=" * 70)
    print("STEP 5: ERROR PATTERN ANALYSIS")
    print("=" * 70)

    # --- With-AI Condition Analysis ---
    print("\n--- With-AI Condition: Override Analysis ---")

    with_ai = responses[responses['condition'] == 'with-ai'].copy()

    # Add AI predictions to responses
    with_ai['ai_prediction'] = with_ai['trial_id'].map(
        lambda x: model_predictions.get(x, {}).get('model_prediction', None)
    )
    with_ai['ai_correct'] = with_ai['ai_prediction'] == with_ai['ground_truth']
    with_ai['human_correct'] = with_ai['participant_answer'] == with_ai['ground_truth']
    with_ai['human_agreed'] = with_ai['participant_answer'] == with_ai['ai_prediction']

    # Categorize trials
    both_correct = with_ai['ai_correct'] & with_ai['human_correct']
    good_override = ~with_ai['ai_correct'] & with_ai['human_correct']  # Human corrected AI
    bad_override = with_ai['ai_correct'] & ~with_ai['human_correct']   # Human overrode correct AI
    both_wrong = ~with_ai['ai_correct'] & ~with_ai['human_correct']

    total = len(with_ai)
    categories = {
        'Both Correct (AI correct, Human correct)': both_correct.sum(),
        'Good Override (AI wrong, Human correct)': good_override.sum(),
        'Bad Override (AI correct, Human wrong)': bad_override.sum(),
        'Both Wrong (AI wrong, Human wrong)': both_wrong.sum()
    }

    print(f"\nTotal with-ai trials: {total}")
    print("\nCategory Breakdown:")
    for cat, count in categories.items():
        pct = count / total * 100
        print(f"  {cat}: {count} ({pct:.1f}%)")

    # Calculate rates
    override_count = (~with_ai['human_agreed']).sum()
    override_rate = override_count / total

    # When human overrode, how often were they right?
    overrides = with_ai[~with_ai['human_agreed']]
    if len(overrides) > 0:
        override_success_rate = overrides['human_correct'].mean()
    else:
        override_success_rate = 0

    agreement_rate = with_ai['human_agreed'].mean()

    print(f"\nKey Metrics:")
    print(f"  Agreement rate (human matched AI): {agreement_rate:.1%}")
    print(f"  Override rate (human disagreed with AI): {override_rate:.1%}")
    print(f"  Override success rate (when disagreed, human was correct): {override_success_rate:.1%}")

    # --- No-AI Condition: Compare to AI error patterns ---
    print("\n--- No-AI Condition: Error Pattern Analysis ---")

    no_ai = responses[responses['condition'] == 'no-ai'].copy()
    no_ai['human_correct'] = no_ai['participant_answer'] == no_ai['ground_truth']

    # Human error rates in no-ai
    human_sarcasm_responses = no_ai[no_ai['ground_truth'] == 'SARCASM']
    human_not_sarcasm_responses = no_ai[no_ai['ground_truth'] == 'NOT_SARCASM']

    human_fn_rate = (human_sarcasm_responses['participant_answer'] == 'NOT_SARCASM').mean()
    human_fp_rate = (human_not_sarcasm_responses['participant_answer'] == 'SARCASM').mean()

    # AI error rates (from A2)
    ai_fn_rate = 1 - AI_ONLY_RECALL  # 1 - 0.872 = 0.128
    ai_fp_rate = 1 - (AI_ONLY_PRECISION * 125 / (AI_ONLY_PRECISION * 125 + (1-AI_ONLY_PRECISION) * 125))
    # Actually: FP / (FP + TN) where FP = 76, TN = 49 from confusion matrix in assignment
    # Let me compute properly: precision = TP / (TP + FP) = 0.645, TP = 109, so FP = 109/0.645 - 109 = 60
    # Recall = TP / (TP + FN) = 0.872, TP = 109, so FN = 109/0.872 - 109 = 16
    # Total = 250, SARCASM = 125, NOT_SARCASM = 125
    # TP = 109, FN = 125 - 109 = 16, FP = precision equation...
    # Let's use known values: accuracy = 174/250 = 0.696
    # If TP + TN = 174, and recall = TP/(TP+FN) = 0.872 for SARCASM
    # TP = 0.872 * 125 = 109, so TN = 174 - 109 = 65
    # FN = 125 - 109 = 16, FP = 125 - 65 = 60
    ai_fp_rate_actual = 60 / 125  # 0.48 (high FP rate)
    ai_fn_rate_actual = 16 / 125  # 0.128 (low FN rate)

    print(f"\nHuman (no-ai condition) error rates:")
    print(f"  False Negative rate (missed sarcasm): {human_fn_rate:.1%}")
    print(f"  False Positive rate (false sarcasm): {human_fp_rate:.1%}")

    print(f"\nAI (from A2) error rates:")
    print(f"  False Negative rate: {ai_fn_rate_actual:.1%}")
    print(f"  False Positive rate: {ai_fp_rate_actual:.1%}")

    print(f"\nComparison:")
    if human_fn_rate > ai_fn_rate_actual:
        print(f"  - Humans miss more sarcasm than AI ({human_fn_rate:.1%} vs {ai_fn_rate_actual:.1%})")
    else:
        print(f"  - AI misses more sarcasm than humans ({ai_fn_rate_actual:.1%} vs {human_fn_rate:.1%})")

    if human_fp_rate > ai_fp_rate_actual:
        print(f"  - Humans have more false positives than AI ({human_fp_rate:.1%} vs {ai_fp_rate_actual:.1%})")
    else:
        print(f"  - AI has more false positives than humans ({ai_fp_rate_actual:.1%} vs {human_fp_rate:.1%})")

    return categories, {
        'agreement_rate': agreement_rate,
        'override_rate': override_rate,
        'override_success_rate': override_success_rate
    }


def generate_charts(responses, participant_stats_all, participant_stats_clean,
                    condition_summary_all, condition_summary_clean, categories):
    """Generate visualization charts."""
    print("\n" + "=" * 70)
    print("STEP 6: GENERATING CHARTS")
    print("=" * 70)

    OUTPUT_DIR.mkdir(exist_ok=True)

    # --- Chart 1: Accuracy by Condition (bar chart with dots) ---
    fig, ax = plt.subplots(figsize=(10, 6))

    conditions = ['no-ai', 'with-ai', 'ai-only']

    # Get mean accuracies
    no_ai_stats = condition_summary_all[condition_summary_all['condition'] == 'no-ai'].iloc[0]
    with_ai_stats = condition_summary_all[condition_summary_all['condition'] == 'with-ai'].iloc[0]

    means = [no_ai_stats['mean_accuracy'], with_ai_stats['mean_accuracy'], AI_ONLY_ACCURACY]
    stds = [no_ai_stats['std_dev'], with_ai_stats['std_dev'], 0]  # AI-only has no std

    x = np.arange(len(conditions))
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7,
                  color=['#3498db', '#2ecc71', '#e74c3c'])

    # Overlay individual participant dots
    no_ai_acc = participant_stats_all[participant_stats_all['condition'] == 'no-ai']['accuracy']
    with_ai_acc = participant_stats_all[participant_stats_all['condition'] == 'with-ai']['accuracy']

    ax.scatter(np.zeros(len(no_ai_acc)) + np.random.normal(0, 0.05, len(no_ai_acc)),
               no_ai_acc, color='darkblue', alpha=0.7, s=50, zorder=5)
    ax.scatter(np.ones(len(with_ai_acc)) + np.random.normal(0, 0.05, len(with_ai_acc)),
               with_ai_acc, color='darkgreen', alpha=0.7, s=50, zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(['No-AI\n(Human Only)', 'With-AI\n(Human+AI)', 'AI-Only'])
    ax.set_ylabel('Accuracy')
    ax.set_title('Sarcasm Detection Accuracy by Condition')
    ax.set_ylim(0, 1)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'accuracy_by_condition.png', dpi=150)
    print(f"  Saved: accuracy_by_condition.png")
    plt.close()

    # --- Chart 2: Per-participant dot plot ---
    fig, ax = plt.subplots(figsize=(10, 6))

    no_ai_df = participant_stats_all[participant_stats_all['condition'] == 'no-ai']
    with_ai_df = participant_stats_all[participant_stats_all['condition'] == 'with-ai']

    # Plot points
    ax.scatter(np.zeros(len(no_ai_df)), no_ai_df['accuracy'],
               s=100, alpha=0.7, label='No-AI')
    ax.scatter(np.ones(len(with_ai_df)), with_ai_df['accuracy'],
               s=100, alpha=0.7, label='With-AI')

    # Add participant labels
    for i, (_, row) in enumerate(no_ai_df.iterrows()):
        short_id = row['worker_id'][-6:]
        ax.annotate(short_id, (0.02, row['accuracy']), fontsize=8)
    for i, (_, row) in enumerate(with_ai_df.iterrows()):
        short_id = row['worker_id'][-6:]
        ax.annotate(short_id, (1.02, row['accuracy']), fontsize=8)

    # Add means
    ax.axhline(y=no_ai_df['accuracy'].mean(), color='blue', linestyle='--', alpha=0.5)
    ax.axhline(y=with_ai_df['accuracy'].mean(), color='orange', linestyle='--', alpha=0.5)
    ax.axhline(y=AI_ONLY_ACCURACY, color='red', linestyle=':', alpha=0.7, label=f'AI-Only ({AI_ONLY_ACCURACY:.1%})')

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['No-AI', 'With-AI'])
    ax.set_ylabel('Accuracy')
    ax.set_title('Per-Participant Accuracy by Condition')
    ax.set_ylim(0, 1)
    ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'participant_accuracy_dotplot.png', dpi=150)
    print(f"  Saved: participant_accuracy_dotplot.png")
    plt.close()

    # --- Chart 3: Override analysis pie chart ---
    fig, ax = plt.subplots(figsize=(8, 8))

    labels = list(categories.keys())
    sizes = list(categories.values())
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#95a5a6']  # Green, Blue, Red, Gray
    explode = (0, 0.1, 0.1, 0)  # Emphasize override categories

    ax.pie(sizes, explode=explode, labels=None, colors=colors, autopct='%1.1f%%',
           shadow=True, startangle=90)
    ax.legend(labels, loc='lower left', fontsize=9)
    ax.set_title('With-AI Condition: Human-AI Agreement Patterns')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'override_analysis_pie.png', dpi=150)
    print(f"  Saved: override_analysis_pie.png")
    plt.close()

    # --- Chart 4: Response time comparison ---
    fig, ax = plt.subplots(figsize=(8, 6))

    # Filter to valid times
    valid_responses = responses[responses['time_spent_ms'] >= 1000].copy()
    valid_responses['time_sec'] = valid_responses['time_spent_ms'] / 1000

    no_ai_times = valid_responses[valid_responses['condition'] == 'no-ai']['time_sec']
    with_ai_times = valid_responses[valid_responses['condition'] == 'with-ai']['time_sec']

    data = [no_ai_times, with_ai_times]
    positions = [0, 1]

    bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True)
    bp['boxes'][0].set_facecolor('#3498db')
    bp['boxes'][1].set_facecolor('#2ecc71')

    ax.set_xticks(positions)
    ax.set_xticklabels(['No-AI', 'With-AI'])
    ax.set_ylabel('Response Time (seconds)')
    ax.set_title('Response Time Distribution by Condition')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'response_time_comparison.png', dpi=150)
    print(f"  Saved: response_time_comparison.png")
    plt.close()

    print(f"\nAll charts saved to {OUTPUT_DIR}/")


def save_csv_outputs(participant_stats_all, participant_stats_clean,
                     condition_summary_all, condition_summary_clean):
    """Save summary tables as CSVs."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    participant_stats_all.to_csv(OUTPUT_DIR / 'participant_accuracy_all.csv', index=False)
    participant_stats_clean.to_csv(OUTPUT_DIR / 'participant_accuracy_clean.csv', index=False)
    condition_summary_all.to_csv(OUTPUT_DIR / 'condition_summary_all.csv', index=False)
    condition_summary_clean.to_csv(OUTPUT_DIR / 'condition_summary_clean.csv', index=False)

    print(f"\nCSV files saved to {OUTPUT_DIR}/")


def print_final_summary(rankings_all, rankings_clean, test_results, categories, override_metrics):
    """Print final summary for report."""
    print("\n" + "=" * 70)
    print("SUMMARY FOR REPORT")
    print("=" * 70)

    print("\n1. ACCURACY RANKINGS")
    print("-" * 40)
    print("\nWith all participants:")
    for i, (name, acc) in enumerate(rankings_all, 1):
        print(f"  {i}. {name}: {acc:.1%}")

    print("\nWithout outliers:")
    for i, (name, acc) in enumerate(rankings_clean, 1):
        print(f"  {i}. {name}: {acc:.1%}")

    print("\n2. STATISTICAL SIGNIFICANCE")
    print("-" * 40)
    sig = "YES" if test_results['p_value'] < 0.05 else "NO"
    print(f"Mann-Whitney U: p = {test_results['p_value']:.4f}, significant at 0.05: {sig}")
    print(f"Effect size (rank-biserial r): {test_results['effect_size']:.3f}")

    print("\n3. A2 PREDICTION EVALUATION")
    print("-" * 40)
    print(f"Prediction: Human-AI Joint > Human-Only > AI-Only")

    # Check if prediction held
    with_ai_rank = next(i for i, (name, _) in enumerate(rankings_clean, 1) if 'With-AI' in name)
    no_ai_rank = next(i for i, (name, _) in enumerate(rankings_clean, 1) if 'No-AI' in name)
    ai_only_rank = next(i for i, (name, _) in enumerate(rankings_clean, 1) if 'AI-Only' in name)

    if with_ai_rank == 1 and no_ai_rank == 2 and ai_only_rank == 3:
        print("Result: PREDICTION CONFIRMED")
    elif with_ai_rank == 1:
        print("Result: PARTIALLY CONFIRMED (With-AI is #1)")
    else:
        print(f"Result: PREDICTION NOT CONFIRMED")
        print(f"  Actual ranking: #{with_ai_rank} With-AI, #{no_ai_rank} No-AI, #{ai_only_rank} AI-Only")

    print("\n4. OVERRIDE ANALYSIS (With-AI Condition)")
    print("-" * 40)
    total = sum(categories.values())
    print(f"Agreement rate: {override_metrics['agreement_rate']:.1%}")
    print(f"Override rate: {override_metrics['override_rate']:.1%}")
    print(f"Override success rate: {override_metrics['override_success_rate']:.1%}")
    print(f"Good overrides: {categories['Good Override (AI wrong, Human correct)']}/{total} "
          f"({categories['Good Override (AI wrong, Human correct)']/total*100:.1f}%)")
    print(f"Bad overrides: {categories['Bad Override (AI correct, Human wrong)']}/{total} "
          f"({categories['Bad Override (AI correct, Human wrong)']/total*100:.1f}%)")

    print("\n5. KEY INSIGHTS")
    print("-" * 40)

    with_ai_acc = rankings_clean[next(i for i, (n, _) in enumerate(rankings_clean) if 'With-AI' in n)][1]
    no_ai_acc = rankings_clean[next(i for i, (n, _) in enumerate(rankings_clean) if 'No-AI' in n)][1]
    improvement = with_ai_acc - no_ai_acc

    if improvement > 0:
        print(f"- AI assistance improved accuracy by {improvement:.1%} ({no_ai_acc:.1%} -> {with_ai_acc:.1%})")
    else:
        print(f"- AI assistance did not improve accuracy ({improvement:.1%} change)")

    if with_ai_acc > AI_ONLY_ACCURACY:
        print(f"- Human-AI collaboration outperformed AI-only ({with_ai_acc:.1%} vs {AI_ONLY_ACCURACY:.1%})")
    else:
        print(f"- Human-AI collaboration did not outperform AI-only")

    if override_metrics['override_success_rate'] > 0.5:
        print(f"- When humans overrode AI, they were usually correct ({override_metrics['override_success_rate']:.1%})")
    else:
        print(f"- When humans overrode AI, they were often wrong ({override_metrics['override_success_rate']:.1%})")


def main():
    """Main analysis pipeline."""
    print("\n" + "=" * 70)
    print("HUMAN-AI SARCASM DETECTION STUDY ANALYSIS")
    print("EECS 543 Assignment 3-2")
    print("=" * 70)

    # Load data
    sessions, responses, model_predictions = load_data()

    # Clean data
    sessions_clean, responses_clean = clean_data(sessions, responses)

    # Performance comparison
    print("\n" + "=" * 70)
    print("STEP 2: PERFORMANCE COMPARISON")
    print("=" * 70)

    # With all participants
    participant_stats_all, condition_summary_all = performance_comparison(responses_clean)
    rankings_all = print_performance_results(participant_stats_all, condition_summary_all,
                                              "(All Participants)")

    # Without outliers
    outliers = [OUTLIER_LOW_ACCURACY, OUTLIER_LOW_EFFORT]
    participant_stats_clean, condition_summary_clean = performance_comparison(
        responses_clean, outliers_to_exclude=outliers
    )
    rankings_clean = print_performance_results(participant_stats_clean, condition_summary_clean,
                                                "(Without Outliers)")

    # Statistical tests
    test_results, perm_results = run_statistical_tests(responses_clean, outliers)

    # Time analysis
    time_stats, participant_time = time_analysis(responses_clean)

    # Error pattern analysis
    categories, override_metrics = error_pattern_analysis(responses_clean, model_predictions)

    # Generate charts
    generate_charts(responses_clean, participant_stats_all, participant_stats_clean,
                    condition_summary_all, condition_summary_clean, categories)

    # Save CSVs
    save_csv_outputs(participant_stats_all, participant_stats_clean,
                     condition_summary_all, condition_summary_clean)

    # Final summary
    print_final_summary(rankings_all, rankings_clean, test_results, categories, override_metrics)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
