"""
Dataset preprocessing for GSM8K mathematical reasoning tasks.
Loads and prepares GSM8K dataset with answer extraction utilities.
"""

import re
from typing import Dict, List, Optional
from datasets import load_dataset
import random


def load_gsm8k(
    split: str = "test",
    num_samples: Optional[int] = None,
    random_seed: int = 42,
    cache_dir: str = ".cache",
) -> List[Dict[str, str]]:
    """
    Load GSM8K dataset.

    Args:
        split: Dataset split ('train' or 'test')
        num_samples: Number of samples to load (None for all)
        random_seed: Random seed for sampling
        cache_dir: Cache directory for HuggingFace datasets

    Returns:
        List of dictionaries with 'question' and 'answer' keys
    """
    # Load GSM8K from HuggingFace
    dataset = load_dataset("gsm8k", "main", split=split, cache_dir=cache_dir)

    # Convert to list of dicts
    problems = []
    for item in dataset:
        # GSM8K format: question field and answer field (with "####" separator)
        question = item["question"]
        answer = item["answer"]

        # Extract numeric answer from "#### <number>" format
        numeric_answer = extract_answer(answer)

        problems.append(
            {
                "question": question,
                "answer": answer,
                "numeric_answer": numeric_answer,
            }
        )

    # Sample if requested
    if num_samples is not None and num_samples < len(problems):
        random.seed(random_seed)
        problems = random.sample(problems, num_samples)

    return problems


def extract_answer(answer_text: str) -> str:
    """
    Extract numeric answer from GSM8K answer format.
    GSM8K answers end with "#### <number>".

    Args:
        answer_text: Full answer text from GSM8K

    Returns:
        Numeric answer as string
    """
    # Look for #### separator
    if "####" in answer_text:
        parts = answer_text.split("####")
        if len(parts) >= 2:
            # Get the numeric part after ####
            numeric = parts[-1].strip()
            # Remove commas and extract number
            numeric = numeric.replace(",", "")
            return numeric

    # Fallback: try to find last number in text
    numbers = re.findall(r"-?\d+\.?\d*", answer_text)
    if numbers:
        return numbers[-1]

    return ""


def extract_prediction(model_output: str) -> str:
    """
    Extract numeric prediction from model output.
    Looks for common answer formats: "The answer is X", "#### X", final number.

    Args:
        model_output: Model's generated text

    Returns:
        Extracted numeric prediction as string
    """
    # Normalize output
    output = model_output.strip()

    # Pattern 1: "#### <number>" (GSM8K format)
    if "####" in output:
        parts = output.split("####")
        if len(parts) >= 2:
            numeric = parts[-1].strip().split()[0]  # Take first token after ####
            numeric = numeric.replace(",", "").rstrip(".")
            return numeric

    # Pattern 2: "The answer is <number>"
    answer_pattern = r"[Tt]he answer is[:\s]+(-?\d+(?:,\d{3})*(?:\.\d+)?)"
    match = re.search(answer_pattern, output)
    if match:
        return match.group(1).replace(",", "")

    # Pattern 3: "= <number>" at end
    equals_pattern = r"=\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)\s*$"
    match = re.search(equals_pattern, output)
    if match:
        return match.group(1).replace(",", "")

    # Pattern 4: Last number in output
    numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", output)
    if numbers:
        return numbers[-1].replace(",", "")

    return ""


def normalize_number(num_str: str) -> float:
    """
    Normalize numeric string to float for comparison.

    Args:
        num_str: Numeric string

    Returns:
        Float value, or None if invalid
    """
    try:
        # Remove commas and convert
        cleaned = num_str.replace(",", "").strip()
        return float(cleaned)
    except (ValueError, AttributeError):
        return None


def answers_match(pred: str, gold: str, tolerance: float = 1e-4) -> bool:
    """
    Check if predicted answer matches gold answer.
    Handles numeric comparison with small tolerance for floating point.

    Args:
        pred: Predicted answer string
        gold: Gold answer string
        tolerance: Numeric tolerance for float comparison

    Returns:
        True if answers match
    """
    # Normalize both
    pred_num = normalize_number(pred)
    gold_num = normalize_number(gold)

    # If both are valid numbers, compare numerically
    if pred_num is not None and gold_num is not None:
        return abs(pred_num - gold_num) < tolerance

    # Otherwise, exact string match (after stripping whitespace)
    return pred.strip() == gold.strip()


def apply_mode_sampling(problems: List[Dict], mode: str) -> List[Dict]:
    """
    Apply mode-specific sampling to dataset.

    Args:
        problems: Full list of problems
        mode: Execution mode ('sanity', 'pilot', or 'full')

    Returns:
        Sampled list of problems
    """
    if mode == "sanity":
        # Sanity: 5-10 samples for quick validation
        return problems[:10] if len(problems) >= 10 else problems
    elif mode == "pilot":
        # Pilot: 20% of dataset (at least 50)
        pilot_size = max(50, int(len(problems) * 0.2))
        return problems[:pilot_size]
    else:  # full
        return problems
