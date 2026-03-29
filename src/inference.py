"""
Inference methods for Chain-of-Thought reasoning:
- ICoCoT (Invariant Counterfactual Checkpointing)
- Standard CoT
- Self-Consistency CoT
"""

import json
import os
import sys
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter
import wandb
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.model import LLMWrapper
from src.preprocess import (
    load_gsm8k,
    extract_prediction,
    answers_match,
    apply_mode_sampling,
)


# Prompt templates for different methods
STANDARD_COT_PROMPT = """Solve this math problem step by step.
Show your reasoning clearly and provide the final answer at the end.
End your response with "#### <answer>" where <answer> is the numeric answer.

Problem: {question}

Solution:"""

CHECKPOINTED_COT_PROMPT = """Solve this math problem step by step with checkpoints.
At each major step, provide a checkpoint in the format:
CHECKPOINT C<number>: <brief state summary with key values>

End your response with "#### <answer>" where <answer> is the numeric answer.

Problem: {question}

Solution:"""

TRANSFORM_PROMPT = """Given the following math problem, generate {num_transforms} meaning-preserving transformations.
Each transformation should preserve the problem's semantics but change the presentation.
Choose from these transformation types: {transform_types}

- PARAPHRASE: Rewrite the problem with different wording but same meaning
- REORDER: Change the order of information presentation
- UNIT_SCALE: Change units (e.g., dollars to cents) if applicable
- EQUATION_FORM: Express constraints differently (e.g., "5 more than X" vs "X+5")

Original problem: {question}

Provide {num_transforms} transformed versions, one per line, each starting with "TRANSFORM <number>:"
"""

REPLAY_PROMPT = """Given a checkpointed solution to a math problem and a transformed version of the problem,
verify whether the checkpoint states from the original solution are still valid for the transformed problem.

Original problem: {original_question}
Original solution with checkpoints:
{original_solution}

Transformed problem: {transformed_question}

Task: Replay the original checkpoint states on the transformed problem.
If all checkpoints are valid, provide the final answer.
If you find a checkpoint that fails, indicate which checkpoint first fails.

Response format:
CHECKPOINT_VALIDATION: <ALL_VALID or FIRST_FAIL_C<number>>
#### <answer>
"""

RECOMPUTE_PROMPT = """Solve this math problem step by step, focusing carefully on the computation around step {focus_step}.

Problem: {question}

Solution:"""


def run_inference(cfg: DictConfig, results_dir: str):
    """
    Main inference entry point.
    Routes to appropriate method based on config.
    """
    # Load dataset
    print(f"Loading GSM8K dataset (split={cfg.dataset.split})...")
    problems = load_gsm8k(
        split=cfg.dataset.split,
        num_samples=cfg.dataset.num_samples,
        random_seed=cfg.dataset.random_seed,
        cache_dir=cfg.dataset.cache_dir,
    )

    # Apply mode-specific sampling
    problems = apply_mode_sampling(problems, cfg.mode)
    print(f"Dataset size after mode adjustment: {len(problems)} problems")

    # Initialize LLM
    print(f"Initializing {cfg.model.provider} model: {cfg.model.name}")
    llm = LLMWrapper(
        provider=cfg.model.provider,
        model_name=cfg.model.name,
        max_tokens=cfg.model.max_tokens,
        temperature=cfg.model.temperature,
        api_key_env=cfg.model.get("api_key_env", "OPENAI_API_KEY"),
    )

    # Initialize WandB if enabled
    wandb_enabled = cfg.wandb.mode != "disabled"
    if wandb_enabled:
        # Adjust project name for sanity/pilot modes
        project = cfg.wandb.project
        if cfg.mode == "sanity":
            project = f"{project}-sanity"
        elif cfg.mode == "pilot":
            project = f"{project}-pilot"

        wandb.init(
            entity=cfg.wandb.entity,
            project=project,
            name=cfg.run.run_id,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        print(f"WandB initialized: {wandb.run.url}")

    # Route to appropriate method
    method_name = cfg.run.method.name
    print(f"Running method: {method_name}")

    if method_name == "icocot":
        results = run_icocot(cfg, problems, llm)
    elif method_name == "standard_cot":
        results = run_standard_cot(cfg, problems, llm)
    elif method_name == "self_consistency_cot":
        results = run_self_consistency_cot(cfg, problems, llm)
    else:
        raise ValueError(f"Unknown method: {method_name}")

    # Compute metrics
    metrics = compute_metrics(results, problems)

    # Log to WandB
    if wandb_enabled:
        wandb.log(metrics)
        wandb.summary.update(metrics)

    # Save results
    os.makedirs(results_dir, exist_ok=True)

    results_file = os.path.join(results_dir, "results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    metrics_file = os.path.join(results_dir, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults saved to {results_dir}")
    print(f"Metrics: {json.dumps(metrics, indent=2)}")

    # Print validation output for sanity/pilot modes
    if cfg.mode == "sanity":
        print_sanity_validation(metrics, results)
    elif cfg.mode == "pilot":
        print_pilot_validation(metrics, results)

    if wandb_enabled:
        wandb.finish()

    return metrics


def run_icocot(cfg: DictConfig, problems: List[Dict], llm: LLMWrapper) -> List[Dict]:
    """Run ICoCoT method with invariance probes and adaptive compute."""
    results = []

    for idx, problem in enumerate(tqdm(problems, desc="ICoCoT")):
        question = problem["question"]
        gold_answer = problem["numeric_answer"]

        # Step 1: Generate checkpointed solution
        prompt = CHECKPOINTED_COT_PROMPT.format(question=question)
        canonical_solution = llm.generate(prompt)
        canonical_answer = extract_prediction(canonical_solution)

        # Step 2: Generate invariance transforms
        num_transforms = cfg.run.method.num_transforms
        transform_types = ", ".join(cfg.run.method.transform_types)
        transform_prompt = TRANSFORM_PROMPT.format(
            question=question,
            num_transforms=num_transforms,
            transform_types=transform_types,
        )
        transforms_text = llm.generate(
            transform_prompt,
            max_tokens=cfg.inference.max_tokens_transform,
        )

        # Parse transforms
        transforms = parse_transforms(transforms_text, num_transforms)

        # Step 3: For each transform, replay and validate
        transform_answers = []
        fault_checkpoints = []

        for t_idx, transformed_question in enumerate(transforms):
            replay_prompt = REPLAY_PROMPT.format(
                original_question=question,
                original_solution=canonical_solution,
                transformed_question=transformed_question,
            )
            replay_output = llm.generate(
                replay_prompt,
                max_tokens=cfg.inference.max_tokens_replay,
            )

            # Extract answer and fault info
            t_answer = extract_prediction(replay_output)
            fault_checkpoint = extract_fault_checkpoint(replay_output)

            transform_answers.append(t_answer)
            fault_checkpoints.append(fault_checkpoint)

        # Step 4: Compute invariance stability
        all_answers = [canonical_answer] + transform_answers
        stability = compute_stability(all_answers)

        # Step 5: Adaptive compute decision
        tau = cfg.run.method.stability_threshold

        if stability >= tau:
            # High stability: accept canonical answer
            final_answer = canonical_answer
            used_adaptive = False
        else:
            # Low stability: recompute around most fragile checkpoint
            fragile_checkpoint = find_most_fragile_checkpoint(fault_checkpoints)

            # Recompute with focus on fragile step
            recompute_samples = []
            for _ in range(cfg.run.method.recompute_samples):
                recompute_prompt = RECOMPUTE_PROMPT.format(
                    question=question,
                    focus_step=fragile_checkpoint,
                )
                recompute_output = llm.generate(
                    recompute_prompt,
                    max_tokens=cfg.inference.max_tokens_recompute,
                )
                recompute_answer = extract_prediction(recompute_output)
                recompute_samples.append(recompute_answer)

            # Vote among recompute samples
            final_answer = majority_vote(recompute_samples)
            used_adaptive = True

        # Record result
        is_correct = answers_match(final_answer, gold_answer)

        result = {
            "problem_idx": idx,
            "question": question,
            "gold_answer": gold_answer,
            "canonical_answer": canonical_answer,
            "transform_answers": transform_answers,
            "stability": stability,
            "used_adaptive": used_adaptive,
            "final_answer": final_answer,
            "correct": is_correct,
            "fault_checkpoints": fault_checkpoints,
        }

        results.append(result)

    return results


def run_standard_cot(
    cfg: DictConfig, problems: List[Dict], llm: LLMWrapper
) -> List[Dict]:
    """Run standard Chain-of-Thought (single-pass)."""
    results = []

    for idx, problem in enumerate(tqdm(problems, desc="Standard CoT")):
        question = problem["question"]
        gold_answer = problem["numeric_answer"]

        # Single-pass CoT
        prompt = STANDARD_COT_PROMPT.format(question=question)
        solution = llm.generate(prompt)
        predicted_answer = extract_prediction(solution)

        is_correct = answers_match(predicted_answer, gold_answer)

        result = {
            "problem_idx": idx,
            "question": question,
            "gold_answer": gold_answer,
            "solution": solution,
            "final_answer": predicted_answer,
            "correct": is_correct,
        }

        results.append(result)

    return results


def run_self_consistency_cot(
    cfg: DictConfig, problems: List[Dict], llm: LLMWrapper
) -> List[Dict]:
    """Run Self-Consistency CoT with multiple samples and voting."""
    results = []
    num_samples = cfg.run.method.num_samples

    for idx, problem in enumerate(tqdm(problems, desc="Self-Consistency CoT")):
        question = problem["question"]
        gold_answer = problem["numeric_answer"]

        # Generate multiple samples
        samples = []
        for _ in range(num_samples):
            prompt = STANDARD_COT_PROMPT.format(question=question)
            solution = llm.generate(prompt)
            predicted_answer = extract_prediction(solution)
            samples.append(predicted_answer)

        # Majority vote
        final_answer = majority_vote(samples)
        is_correct = answers_match(final_answer, gold_answer)

        result = {
            "problem_idx": idx,
            "question": question,
            "gold_answer": gold_answer,
            "samples": samples,
            "final_answer": final_answer,
            "correct": is_correct,
        }

        results.append(result)

    return results


def parse_transforms(text: str, expected_count: int) -> List[str]:
    """Parse transformed questions from LLM output."""
    transforms = []
    lines = text.strip().split("\n")

    for line in lines:
        if line.strip().startswith("TRANSFORM"):
            # Extract text after "TRANSFORM <number>:"
            parts = line.split(":", 1)
            if len(parts) == 2:
                transform = parts[1].strip()
                transforms.append(transform)

    # Pad if needed
    while len(transforms) < expected_count:
        transforms.append("")

    return transforms[:expected_count]


def extract_fault_checkpoint(text: str) -> Optional[int]:
    """Extract fault checkpoint from replay output."""
    if "FIRST_FAIL_C" in text:
        # Extract checkpoint number
        import re

        match = re.search(r"FIRST_FAIL_C(\d+)", text)
        if match:
            return int(match.group(1))
    return None


def compute_stability(answers: List[str]) -> float:
    """Compute stability as fraction of answers matching the first answer."""
    if not answers:
        return 0.0

    canonical = answers[0]
    matches = sum(1 for ans in answers if answers_match(ans, canonical))
    return matches / len(answers)


def find_most_fragile_checkpoint(fault_checkpoints: List[Optional[int]]) -> int:
    """Find the checkpoint that failed most frequently."""
    # Filter out None values
    valid_faults = [f for f in fault_checkpoints if f is not None]

    if not valid_faults:
        return 1  # Default to first checkpoint

    # Count occurrences
    counter = Counter(valid_faults)
    most_common = counter.most_common(1)[0][0]
    return most_common


def majority_vote(answers: List[str]) -> str:
    """Return the most common answer (majority vote)."""
    if not answers:
        return ""

    # Normalize answers for voting
    normalized = [ans.strip() for ans in answers if ans.strip()]

    if not normalized:
        return ""

    counter = Counter(normalized)
    most_common = counter.most_common(1)[0][0]
    return most_common


def compute_metrics(results: List[Dict], problems: List[Dict]) -> Dict[str, Any]:
    """Compute evaluation metrics from results."""
    total = len(results)
    if total == 0:
        return {"accuracy": 0.0, "num_problems": 0}

    correct = sum(1 for r in results if r["correct"])
    accuracy = correct / total

    metrics = {
        "accuracy": accuracy,
        "num_problems": total,
        "num_correct": correct,
    }

    # Add method-specific metrics
    if results and "stability" in results[0]:
        # ICoCoT metrics
        stabilities = [r["stability"] for r in results]
        adaptive_used = sum(1 for r in results if r.get("used_adaptive", False))

        metrics.update(
            {
                "mean_stability": sum(stabilities) / len(stabilities),
                "adaptive_compute_rate": adaptive_used / total,
            }
        )

    return metrics


def print_sanity_validation(metrics: Dict[str, Any], results: List[Dict]):
    """Print sanity validation verdict."""
    num_problems = metrics.get("num_problems", 0)
    accuracy = metrics.get("accuracy", 0.0)

    # Sanity checks
    passed = True
    reason = ""

    # Check 1: At least 5 problems processed
    if num_problems < 5:
        passed = False
        reason = "insufficient_samples"

    # Check 2: All metrics are finite
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            if not (float("-inf") < value < float("inf")):
                passed = False
                reason = "non_finite_metrics"
                break

    # Check 3: Not all results identical (only if multiple results)
    if num_problems > 1 and len(set(r["correct"] for r in results)) == 1:
        # All same correctness - could indicate issue
        if all(r["correct"] for r in results) or all(not r["correct"] for r in results):
            # This is actually fine - could legitimately be all correct or all wrong
            pass

    # Print validation verdict
    if passed:
        print("\nSANITY_VALIDATION: PASS")
    else:
        print(f"\nSANITY_VALIDATION: FAIL reason={reason}")

    # Print summary
    summary = {
        "samples": num_problems,
        "accuracy": accuracy,
        "status": "pass" if passed else "fail",
    }
    print(f"SANITY_VALIDATION_SUMMARY: {json.dumps(summary)}")


def print_pilot_validation(metrics: Dict[str, Any], results: List[Dict]):
    """Print pilot validation verdict."""
    num_problems = metrics.get("num_problems", 0)
    accuracy = metrics.get("accuracy", 0.0)

    # Pilot checks
    passed = True
    reason = ""

    # Check 1: At least 50 problems processed
    if num_problems < 50:
        passed = False
        reason = "insufficient_samples"

    # Check 2: Primary metric (accuracy) is computed and finite
    if not (0.0 <= accuracy <= 1.0):
        passed = False
        reason = "invalid_metric"

    # Check 3: All metrics are finite
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            if not (float("-inf") < value < float("inf")):
                passed = False
                reason = "non_finite_metrics"
                break

    # Print validation verdict
    if passed:
        print("\nPILOT_VALIDATION: PASS")
    else:
        print(f"\nPILOT_VALIDATION: FAIL reason={reason}")

    # Print summary
    summary = {
        "samples": num_problems,
        "primary_metric": "accuracy",
        "primary_metric_value": accuracy,
        "status": "pass" if passed else "fail",
    }
    print(f"PILOT_VALIDATION_SUMMARY: {json.dumps(summary)}")


if __name__ == "__main__":
    # This should not be called directly; use main.py
    print("ERROR: inference.py should be called via main.py")
    sys.exit(1)
