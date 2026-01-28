import os
import subprocess
import sys

import hydra
from hydra.utils import get_original_cwd


def apply_mode_overrides(cfg):
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        if hasattr(cfg, "run") and getattr(cfg.run, "optuna", None) is not None:
            cfg.run.optuna.n_trials = 0
    elif cfg.mode == "full":
        cfg.wandb.mode = "online"
    return cfg


def build_overrides(cfg) -> list:
    run_id = cfg.run.run_id if hasattr(cfg, "run") else cfg.run
    overrides = [
        f"runs@run={run_id}",
        f"results_dir={cfg.results_dir}",
        f"mode={cfg.mode}",
    ]
    if hasattr(cfg, "wandb"):
        overrides.append(f"wandb.mode={cfg.wandb.mode}")
    if hasattr(cfg, "run") and getattr(cfg.run, "optuna", None) is not None:
        overrides.append(f"run.optuna.n_trials={int(cfg.run.optuna.n_trials)}")
    return overrides


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg):
    root = get_original_cwd()
    if not hasattr(cfg, "run") or cfg.run is None:
        raise ValueError("run=<run_id> must be specified.")
    if not hasattr(cfg.run, "run_id"):
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
