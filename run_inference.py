"""
Sarcasm Detection - AI Model Inference Script
==============================================
Author: Nikhil Guna
Course: EECS 543 - AI Ethics
Assignment 2

This script runs Claude Haiku 4.5 on the curated study dataset to generate
sarcasm predictions with confidence scores and reasoning.

Setup:
    pip install anthropic

Usage:
    python run_inference.py

    The script saves progress every 25 trials and can resume from where it left off.
    Output: model_outputs.json

Configuration:
    - Set ANTHROPIC_API_KEY below or as environment variable
    - Model: claude-haiku-4-5-20251001 (Anthropic)
"""

import anthropic
import json
import time
import re
import os

# ============ CONFIGURATION ============
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "YOUR_API_KEY_HERE")
MODEL = "claude-haiku-4-5-20251001"
INPUT_FILE = "study_dataset.json"
OUTPUT_FILE = "model_outputs.json"
DELAY_BETWEEN_CALLS = 0.5  # seconds
# ========================================

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def build_prompt(trial):
    """Build the sarcasm detection prompt for a single trial."""
    context_str = "\n".join(
        [f"[Comment {i+1}]: {c}" for i, c in enumerate(trial['context'])]
    )
    return f"""Detect sarcasm. Given this Reddit thread context and response, classify the response.

Context:
{context_str}

Response: "{trial['response']}"

Reply with ONLY this JSON (no markdown):
{{"prediction": "SARCASM" or "NOT_SARCASM", "confidence": <0-100>, "reasoning": "<1-2 sentences>"}}"""


def parse_output(text):
    """Parse the model's JSON response, handling common formatting issues."""
    text = re.sub(r'```json\s*|```', '', text).strip()
    try:
        result = json.loads(text)
        pred = result.get('prediction', '').upper()
        result['prediction'] = 'NOT_SARCASM' if 'NOT' in pred else 'SARCASM'
        result['confidence'] = int(result.get('confidence', 50))
        result['reasoning'] = result.get('reasoning', 'No reasoning provided')
        return result
    except json.JSONDecodeError:
        # Fallback: extract prediction from raw text
        text_upper = text.upper()
        if 'NOT_SARCASM' in text_upper or 'NOT SARCASM' in text_upper:
            pred = 'NOT_SARCASM'
        elif 'SARCASM' in text_upper:
            pred = 'SARCASM'
        else:
            pred = 'UNKNOWN'
        return {
            'prediction': pred,
            'confidence': 50,
            'reasoning': f'Parse error. Raw: {text[:200]}'
        }


def run_inference():
    """Run model inference on all trials with resume support."""
    with open(INPUT_FILE) as f:
        trials = json.load(f)
    print(f"Loaded {len(trials)} trials")

    # Resume from saved progress
    results = []
    start_idx = 0
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as f:
            results = json.load(f)
        start_idx = len(results)
        print(f"Resuming from trial {start_idx + 1}")

    correct_count = sum(
        1 for r in results if r['model_prediction'] == r['ground_truth']
    )

    for i in range(start_idx, len(trials)):
        trial = trials[i]
        prompt = build_prompt(trial)

        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}]
            )
            output = parse_output(resp.content[0].text)
        except Exception as e:
            print(f"  ERROR trial {trial['trial_id']}: {e}")
            output = {
                'prediction': 'ERROR',
                'confidence': 0,
                'reasoning': str(e)
            }
            time.sleep(3)

        result = {
            'trial_id': trial['trial_id'],
            'context': trial['context'],
            'response': trial['response'],
            'ground_truth': trial['label'],
            'model_prediction': output['prediction'],
            'model_confidence': output['confidence'],
            'model_reasoning': output.get('reasoning', '')
        }
        results.append(result)

        is_correct = output['prediction'] == trial['label']
        if is_correct:
            correct_count += 1
        mark = '✓' if is_correct else '✗'
        acc = correct_count / (i + 1) * 100
        print(
            f"[{i+1}/{len(trials)}] {mark} "
            f"GT={trial['label']:13s} "
            f"Pred={output['prediction']:13s} "
            f"Conf={output['confidence']}% "
            f"Acc={acc:.1f}%"
        )

        # Save progress
        if (i + 1) % 25 == 0:
            with open(OUTPUT_FILE, 'w') as f:
                json.dump(results, f, indent=2)

        time.sleep(DELAY_BETWEEN_CALLS)

    # Final save
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)

    valid = [r for r in results if r['model_prediction'] != 'ERROR']
    correct = sum(1 for r in valid if r['model_prediction'] == r['ground_truth'])
    print(f"\nDONE! Accuracy: {correct}/{len(valid)} = {correct/len(valid)*100:.1f}%")


if __name__ == '__main__':
    run_inference()
