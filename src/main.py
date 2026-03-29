"""
Main orchestrator for ICoCoT experiments.
Handles Hydra configuration and mode-specific parameter overrides.
"""

import os
import sys
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf

from src.inference import run_inference


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    """
    Main entry point for running a single experiment.

    Args:
        cfg: Hydra configuration
    """
    print("=" * 80)
    print(f"ICoCoT Experiment: {cfg.run.run_id}")
    print(f"Mode: {cfg.mode}")
    print("=" * 80)

    # Apply mode-specific overrides
    cfg = apply_mode_overrides(cfg)

    # Print effective configuration
    print("\nEffective Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    # Prepare results directory
    results_dir = os.path.join(cfg.results_dir, cfg.run.run_id)
    os.makedirs(results_dir, exist_ok=True)

    # Run inference
    try:
        metrics = run_inference(cfg, results_dir)

        print("\n" + "=" * 80)
        print(f"Experiment completed successfully!")
        print(f"Results saved to: {results_dir}")
        print(f"Final accuracy: {metrics.get('accuracy', 0.0):.4f}")
        print("=" * 80)

        return 0

    except Exception as e:
        print(f"\n{'=' * 80}")
        print(f"ERROR: Experiment failed with exception:")
        print(f"{type(e).__name__}: {e}")
        print("=" * 80)
        import traceback

        traceback.print_exc()
        return 1


def apply_mode_overrides(cfg: DictConfig) -> DictConfig:
    """
    Apply mode-specific parameter overrides.

    Args:
        cfg: Original configuration

    Returns:
        Updated configuration with mode overrides
    """
    mode = cfg.mode

    if mode == "sanity":
        print("Applying SANITY mode overrides...")

        # Adjust dataset size
        with_override = OmegaConf.to_container(cfg, resolve=True)
        # Sanity mode handled in apply_mode_sampling function (10 samples)

        # Keep WandB online for sanity
        cfg.wandb.mode = "online"

        # No optuna trials in sanity mode (if optuna config exists)
        if "optuna" in cfg:
            cfg.optuna.n_trials = 0

        print(f"  - Dataset will be limited to ~10 samples")
        print(f"  - WandB: {cfg.wandb.mode}")

    elif mode == "pilot":
        print("Applying PILOT mode overrides...")

        # Pilot mode handled in apply_mode_sampling function (20% of data, min 50)

        # Keep WandB online
        cfg.wandb.mode = "online"

        # Reduce optuna trials if present
        if "optuna" in cfg:
            cfg.optuna.n_trials = 3

        print(f"  - Dataset will be limited to ~20% (min 50 samples)")
        print(f"  - WandB: {cfg.wandb.mode}")

    elif mode == "full":
        print("Running in FULL mode (no overrides)")
        # No overrides for full mode
        pass

    else:
        print(f"WARNING: Unknown mode '{mode}', treating as 'full'")

    return cfg


if __name__ == "__main__":
    sys.exit(main())
