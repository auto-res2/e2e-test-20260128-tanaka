import os
import subprocess
import sys

import hydra
from hydra.utils import get_original_cwd


def apply_mode_overrides(cfg):
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        if hasattr(cfg, "run") and not isinstance(cfg.run, str) and getattr(cfg.run, "optuna", None) is not None:
            cfg.run.optuna.n_trials = 0
    elif cfg.mode == "full":
        cfg.wandb.mode = "online"
    return cfg


def build_overrides(cfg) -> list:
    # Handle both cases: cfg.run as string or as config object
    if isinstance(cfg.run, str):
        run_id = cfg.run
    else:
        run_id = cfg.run.run_id if hasattr(cfg.run, "run_id") else str(cfg.run)
    
    overrides = [
        f"runs@run={run_id}",
        f"results_dir={cfg.results_dir}",
        f"mode={cfg.mode}",
    ]
    if hasattr(cfg, "wandb"):
        overrides.append(f"wandb.mode={cfg.wandb.mode}")
    if hasattr(cfg, "run") and not isinstance(cfg.run, str) and getattr(cfg.run, "optuna", None) is not None:
        overrides.append(f"run.optuna.n_trials={int(cfg.run.optuna.n_trials)}")
    return overrides


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg):
    root = get_original_cwd()
    if not hasattr(cfg, "run") or cfg.run is None:
        raise ValueError("run=<run_id> must be specified.")
    # cfg.run can be either a string (run_id) or a config object with run_id field
    if isinstance(cfg.run, str):
        # If run is just a string, it's the run_id itself
        pass
    elif not hasattr(cfg.run, "run_id"):
        raise ValueError("Run configuration must include run_id.")

    cfg = apply_mode_overrides(cfg)

    if not os.path.isabs(cfg.results_dir):
        cfg.results_dir = os.path.abspath(os.path.join(root, cfg.results_dir))
    os.makedirs(cfg.results_dir, exist_ok=True)

    overrides = build_overrides(cfg)
    cmd = [sys.executable, "-u", "-m", "src.train"] + overrides
    subprocess.run(cmd, check=True, cwd=root)


if __name__ == "__main__":
    main()
