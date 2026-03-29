"""
Evaluation script for comparing multiple runs.
Fetches results from WandB and generates comparison metrics and plots.
"""

import argparse
import json
import os
from typing import List, Dict, Any
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def main():
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(description="Evaluate and compare experiment runs")
    parser.add_argument(
        "--results_dir", type=str, required=True, help="Results directory"
    )
    parser.add_argument(
        "--run_ids", type=str, required=True, help="JSON list of run IDs to compare"
    )
    parser.add_argument(
        "--wandb_entity", type=str, default="airas", help="WandB entity"
    )
    parser.add_argument(
        "--wandb_project", type=str, default="20260329-153353", help="WandB project"
    )

    args = parser.parse_args()

    # Parse run IDs
    run_ids = json.loads(args.run_ids)
    print(f"Evaluating {len(run_ids)} runs: {run_ids}")

    # Initialize WandB API
    api = wandb.Api()

    # Fetch data for each run
    run_data = {}
    for run_id in run_ids:
        print(f"\nFetching data for {run_id}...")
        data = fetch_run_data(api, args.wandb_entity, args.wandb_project, run_id)
        run_data[run_id] = data

        # Export per-run metrics
        run_dir = os.path.join(args.results_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)

        metrics_file = os.path.join(run_dir, "metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(data["summary"], f, indent=2)
        print(f"  Saved metrics to {metrics_file}")

        # Generate per-run figures
        generate_run_figures(data, run_dir, run_id)

    # Generate comparison metrics and plots
    comparison_dir = os.path.join(args.results_dir, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)

    # Aggregate metrics
    aggregated = aggregate_metrics(run_data, run_ids)

    agg_file = os.path.join(comparison_dir, "aggregated_metrics.json")
    with open(agg_file, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"\nSaved aggregated metrics to {agg_file}")

    # Generate comparison plots
    generate_comparison_plots(run_data, run_ids, comparison_dir)

    print("\nEvaluation complete!")
    print(f"Results saved to {args.results_dir}")


def fetch_run_data(
    api: wandb.Api, entity: str, project: str, run_id: str
) -> Dict[str, Any]:
    """
    Fetch run data from WandB by display name.

    Args:
        api: WandB API instance
        entity: WandB entity
        project: WandB project
        run_id: Run display name

    Returns:
        Dictionary with run config, summary, and history
    """
    # Try main project first
    runs = api.runs(
        f"{entity}/{project}", filters={"display_name": run_id}, order="-created_at"
    )

    # If not found, try sanity/pilot projects
    if len(runs) == 0:
        for suffix in ["-sanity", "-pilot"]:
            runs = api.runs(
                f"{entity}/{project}{suffix}",
                filters={"display_name": run_id},
                order="-created_at",
            )
            if len(runs) > 0:
                break

    if len(runs) == 0:
        raise ValueError(f"No run found with display name: {run_id}")

    run = runs[0]  # Most recent run with that name

    # Fetch data
    config = run.config
    summary = dict(run.summary)
    history = run.history()

    return {
        "config": config,
        "summary": summary,
        "history": history,
        "run": run,
    }


def generate_run_figures(data: Dict[str, Any], run_dir: str, run_id: str):
    """Generate per-run figures."""
    summary = data["summary"]

    # Figure 1: Metrics bar chart
    fig, ax = plt.subplots(figsize=(8, 6))

    metrics_to_plot = {
        k: v for k, v in summary.items() if isinstance(v, (int, float)) and k != "_step"
    }

    if metrics_to_plot:
        metric_names = list(metrics_to_plot.keys())
        metric_values = list(metrics_to_plot.values())

        ax.barh(metric_names, metric_values)
        ax.set_xlabel("Value")
        ax.set_title(f"Metrics for {run_id}")
        ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        fig_path = os.path.join(run_dir, "metrics_summary.pdf")
        plt.savefig(fig_path, format="pdf", bbox_inches="tight")
        plt.close()
        print(f"  Generated {fig_path}")


def aggregate_metrics(run_data: Dict[str, Dict], run_ids: List[str]) -> Dict[str, Any]:
    """
    Aggregate metrics across runs.

    Args:
        run_data: Dictionary mapping run_id to run data
        run_ids: List of run IDs

    Returns:
        Aggregated metrics with comparison
    """
    # Collect metrics by run
    metrics_by_run = {}

    for run_id in run_ids:
        summary = run_data[run_id]["summary"]
        metrics_by_run[run_id] = {
            "accuracy": summary.get("accuracy", 0.0),
            "num_problems": summary.get("num_problems", 0),
            "num_correct": summary.get("num_correct", 0),
        }

        # Add method-specific metrics if available
        if "mean_stability" in summary:
            metrics_by_run[run_id]["mean_stability"] = summary["mean_stability"]
        if "adaptive_compute_rate" in summary:
            metrics_by_run[run_id]["adaptive_compute_rate"] = summary[
                "adaptive_compute_rate"
            ]

    # Identify proposed vs baseline runs
    proposed_runs = [rid for rid in run_ids if "proposed" in rid]
    baseline_runs = [rid for rid in run_ids if "comparative" in rid]

    # Find best in each category
    best_proposed = None
    best_proposed_acc = -1.0
    if proposed_runs:
        for rid in proposed_runs:
            acc = metrics_by_run[rid]["accuracy"]
            if acc > best_proposed_acc:
                best_proposed = rid
                best_proposed_acc = acc

    best_baseline = None
    best_baseline_acc = -1.0
    if baseline_runs:
        for rid in baseline_runs:
            acc = metrics_by_run[rid]["accuracy"]
            if acc > best_baseline_acc:
                best_baseline = rid
                best_baseline_acc = acc

    # Compute gap
    gap = None
    if best_proposed is not None and best_baseline is not None:
        gap = best_proposed_acc - best_baseline_acc

    aggregated = {
        "primary_metric": "accuracy",
        "metrics_by_run": metrics_by_run,
        "best_proposed": best_proposed,
        "best_proposed_accuracy": best_proposed_acc if best_proposed else None,
        "best_baseline": best_baseline,
        "best_baseline_accuracy": best_baseline_acc if best_baseline else None,
        "gap": gap,
    }

    return aggregated


def generate_comparison_plots(
    run_data: Dict[str, Dict], run_ids: List[str], output_dir: str
):
    """Generate comparison plots across runs."""

    # Set style
    sns.set_style("whitegrid")

    # Figure 1: Accuracy comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    accuracies = []
    labels = []
    colors = []

    for run_id in run_ids:
        summary = run_data[run_id]["summary"]
        acc = summary.get("accuracy", 0.0)
        accuracies.append(acc)
        labels.append(run_id)

        # Color by type
        if "proposed" in run_id:
            colors.append("#2ecc71")  # Green for proposed
        else:
            colors.append("#3498db")  # Blue for baselines

    bars = ax.barh(labels, accuracies, color=colors)
    ax.set_xlabel("Accuracy", fontsize=12)
    ax.set_title("Accuracy Comparison Across Methods", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1.0)
    ax.grid(axis="x", alpha=0.3)

    # Add value labels
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        ax.text(acc + 0.01, i, f"{acc:.3f}", va="center", fontsize=10)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "comparison_accuracy.pdf")
    plt.savefig(fig_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Generated {fig_path}")

    # Figure 2: Method-specific metrics (if available)
    # Check if any run has stability metrics (ICoCoT)
    has_stability = any("mean_stability" in run_data[rid]["summary"] for rid in run_ids)

    if has_stability:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Subplot 1: Stability
        stabilities = []
        stability_labels = []
        for run_id in run_ids:
            summary = run_data[run_id]["summary"]
            if "mean_stability" in summary:
                stabilities.append(summary["mean_stability"])
                stability_labels.append(run_id)

        if stabilities:
            ax1.barh(stability_labels, stabilities, color="#e74c3c")
            ax1.set_xlabel("Mean Invariance Stability", fontsize=12)
            ax1.set_title(
                "Invariance Stability (ICoCoT)", fontsize=12, fontweight="bold"
            )
            ax1.set_xlim(0, 1.0)
            ax1.grid(axis="x", alpha=0.3)

        # Subplot 2: Adaptive compute rate
        adaptive_rates = []
        adaptive_labels = []
        for run_id in run_ids:
            summary = run_data[run_id]["summary"]
            if "adaptive_compute_rate" in summary:
                adaptive_rates.append(summary["adaptive_compute_rate"])
                adaptive_labels.append(run_id)

        if adaptive_rates:
            ax2.barh(adaptive_labels, adaptive_rates, color="#9b59b6")
            ax2.set_xlabel("Adaptive Compute Rate", fontsize=12)
            ax2.set_title("Adaptive Compute Usage", fontsize=12, fontweight="bold")
            ax2.set_xlim(0, 1.0)
            ax2.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        fig_path = os.path.join(output_dir, "comparison_icocot_metrics.pdf")
        plt.savefig(fig_path, format="pdf", bbox_inches="tight")
        plt.close()
        print(f"Generated {fig_path}")

    # Figure 3: Summary table
    fig, ax = plt.subplots(figsize=(12, max(4, len(run_ids) * 0.6)))
    ax.axis("tight")
    ax.axis("off")

    # Create table data
    table_data = []
    headers = ["Method", "Accuracy", "Correct", "Total"]

    for run_id in run_ids:
        summary = run_data[run_id]["summary"]
        row = [
            run_id,
            f"{summary.get('accuracy', 0.0):.4f}",
            str(summary.get("num_correct", 0)),
            str(summary.get("num_problems", 0)),
        ]
        table_data.append(row)

    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc="center",
        loc="center",
        colWidths=[0.4, 0.2, 0.2, 0.2],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor("#34495e")
        cell.set_text_props(weight="bold", color="white")

    # Style rows
    for i in range(len(table_data)):
        for j in range(len(headers)):
            cell = table[(i + 1, j)]
            if i % 2 == 0:
                cell.set_facecolor("#ecf0f1")

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "comparison_table.pdf")
    plt.savefig(fig_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Generated {fig_path}")


if __name__ == "__main__":
    main()
