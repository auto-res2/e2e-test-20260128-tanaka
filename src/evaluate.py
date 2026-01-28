import json
import os
import sys
from typing import Any, Dict, List, Tuple

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from omegaconf import OmegaConf
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TARGET_STEP = 2000


def parse_cli_args(argv: List[str]) -> Dict[str, str]:
    args: Dict[str, str] = {}
    for item in argv:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        args[key] = value
    if "results_dir" not in args or "run_ids" not in args:
        raise ValueError(
            "Usage: python -m src.evaluate results_dir=... run_ids='[\"run-1\", \"run-2\"]'"
        )
    return args


def to_jsonable(value):
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (np.generic, np.ndarray)):
        return np.asarray(value).tolist()
    return value


def load_wandb_config(root_dir: str) -> Dict[str, Any]:
    cfg = OmegaConf.load(os.path.join(root_dir, "config", "config.yaml"))
    return {
        "entity": cfg.wandb.entity,
        "project": cfg.wandb.project,
    }


def fetch_history(run: wandb.apis.public.Run) -> pd.DataFrame:
    history = run.history(samples=None, pandas=True)
    if history is None:
        history = pd.DataFrame()
    if history.empty:
        rows: List[Dict[str, Any]] = []
        try:
            for row in run.scan_history():
                rows.append(row)
        except Exception:
            rows = []
        if rows:
            history = pd.DataFrame(rows)

    if "_step" not in history.columns:
        if "step" in history.columns:
            history["_step"] = history["step"]
        else:
            history["_step"] = np.arange(len(history))
    return history


def ensure_not_trial_run(config: Dict[str, Any], run_id: str):
    mode = config.get("mode")
    wandb_mode = None
    if isinstance(config.get("wandb"), dict):
        wandb_mode = config.get("wandb", {}).get("mode")
    if config.get("wandb.mode") is not None:
        wandb_mode = config.get("wandb.mode")
    if mode == "trial" or str(wandb_mode).lower() == "disabled":
        raise ValueError(
            f"Run {run_id} appears to be a trial/disabled WandB run; refusing evaluation."
        )


def save_json(path: str, payload: Dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def select_step_col(history: pd.DataFrame) -> str:
    if "combo_step" in history.columns:
        return "combo_step"
    return "_step"


def metric_series(history: pd.DataFrame, metric: str) -> Tuple[np.ndarray, np.ndarray]:
    if metric not in history:
        return np.array([]), np.array([])
    step_col = select_step_col(history)
    df = history.dropna(subset=[metric])
    if df.empty:
        return np.array([]), np.array([])
    if "combo_id" in df.columns:
        grouped = df.groupby(step_col)[metric].mean().reset_index()
        return grouped[step_col].to_numpy(), grouped[metric].to_numpy()
    df = df.sort_values(step_col)
    return df[step_col].to_numpy(), df[metric].to_numpy()


def metric_at_step_by_combo(history: pd.DataFrame, metric: str, target_step: int) -> List[float]:
    if metric not in history:
        return []
    step_col = select_step_col(history)
    df = history.dropna(subset=[metric])
    if df.empty:
        return []
    if "combo_id" in df.columns:
        values = []
        for _, group in df.groupby("combo_id"):
            group = group.sort_values(step_col)
            match = group.loc[group[step_col] == target_step, metric]
            if match.empty:
                value = group[metric].iloc[-1]
            else:
                value = match.iloc[-1]
            values.append(float(value))
        return values
    group = df.sort_values(step_col)
    match = group.loc[group[step_col] == target_step, metric]
    value = match.iloc[-1] if not match.empty else group[metric].iloc[-1]
    return [float(value)]


def final_metric_by_combo(history: pd.DataFrame, metric: str) -> List[float]:
    if metric not in history:
        return []
    step_col = select_step_col(history)
    df = history.dropna(subset=[metric])
    if df.empty:
        return []
    if "combo_id" in df.columns:
        df = df.sort_values(step_col)
        return df.groupby("combo_id").tail(1)[metric].astype(float).tolist()
    return [float(df.sort_values(step_col)[metric].iloc[-1])]


def median_or_none(values: List[float]) -> float:
    if not values:
        return None
    return float(np.median(values))


def plot_learning_curves(history: pd.DataFrame, run_id: str, out_dir: str) -> List[str]:
    paths = []
    train_steps, train_vals = metric_series(history, "train_loss")
    val_steps, val_vals = metric_series(history, "val_perplexity")
    if len(train_vals) == 0 and len(val_vals) == 0:
        return paths

    plt.figure(figsize=(8, 5))
    if len(train_vals) > 0:
        plt.plot(train_steps, train_vals, label="train_loss")
    if len(val_vals) > 0:
        plt.plot(val_steps, val_vals, label="val_perplexity")
        best_idx = int(np.argmin(val_vals))
        plt.scatter(val_steps[best_idx], val_vals[best_idx], color="red", zorder=3)
        plt.annotate(
            f"best={val_vals[best_idx]:.2f}",
            (val_steps[best_idx], val_vals[best_idx]),
            textcoords="offset points",
            xytext=(5, 5),
        )
        plt.scatter(val_steps[-1], val_vals[-1], color="black", zorder=3)
        plt.annotate(
            f"final={val_vals[-1]:.2f}",
            (val_steps[-1], val_vals[-1]),
            textcoords="offset points",
            xytext=(5, -10),
        )
    plt.xlabel("step")
    plt.ylabel("metric")
    plt.title(f"Learning Curves: {run_id}")
    plt.legend()
    plt.tight_layout()

    fig_path = os.path.join(out_dir, f"{run_id}_learning_curve.pdf")
    plt.savefig(fig_path)
    plt.close()
    paths.append(fig_path)
    return paths


def plot_rank_lambda(history: pd.DataFrame, run_id: str, out_dir: str) -> List[str]:
    paths = []
    rank_steps, rank_vals = metric_series(history, "mean_r_eff")
    lam_steps, lam_vals = metric_series(history, "lambda")
    if len(rank_vals) == 0 and len(lam_vals) == 0:
        return paths

    plt.figure(figsize=(8, 5))
    if len(rank_vals) > 0:
        plt.plot(rank_steps, rank_vals, label="mean_r_eff")
    if len(lam_vals) > 0:
        plt.plot(lam_steps, lam_vals, label="lambda")
    plt.xlabel("step")
    plt.ylabel("value")
    plt.title(f"Rank/λ Dynamics: {run_id}")
    plt.legend()
    plt.tight_layout()

    fig_path = os.path.join(out_dir, f"{run_id}_rank_lambda_curve.pdf")
    plt.savefig(fig_path)
    plt.close()
    paths.append(fig_path)
    return paths


def plot_scatter(history: pd.DataFrame, run_id: str, out_dir: str) -> List[str]:
    paths = []
    if "mean_r_eff" not in history or "val_perplexity" not in history:
        return paths
    df = history.dropna(subset=["mean_r_eff", "val_perplexity"])
    if df.empty:
        return paths
    plt.figure(figsize=(6, 5))
    plt.scatter(df["mean_r_eff"], df["val_perplexity"], alpha=0.6)
    plt.xlabel("mean_r_eff")
    plt.ylabel("val_perplexity")
    plt.title(f"Rank vs Perplexity: {run_id}")
    plt.tight_layout()

    fig_path = os.path.join(out_dir, f"{run_id}_rank_vs_perplexity_scatter.pdf")
    plt.savefig(fig_path)
    plt.close()
    paths.append(fig_path)
    return paths


def plot_rank_histogram(history: pd.DataFrame, run_id: str, out_dir: str) -> List[str]:
    paths = []
    if "mean_r_eff" not in history:
        return paths
    df = history.dropna(subset=["mean_r_eff"])
    if df.empty:
        return paths
    plt.figure(figsize=(6, 4))
    sns.histplot(df["mean_r_eff"], kde=True)
    plt.xlabel("mean_r_eff")
    plt.title(f"Mean Effective Rank Histogram: {run_id}")
    plt.tight_layout()

    fig_path = os.path.join(out_dir, f"{run_id}_rank_histogram.pdf")
    plt.savefig(fig_path)
    plt.close()
    paths.append(fig_path)
    return paths


def infer_eval_interval(steps: np.ndarray) -> float:
    uniq = np.unique(steps)
    if len(uniq) < 2:
        return 1.0
    return float(np.median(np.diff(uniq)))


def compute_steps_to_target(history: pd.DataFrame, target: float) -> Tuple[List[float], float]:
    if "val_perplexity" not in history:
        return [], 0.0
    step_col = select_step_col(history)
    step_values = []
    df = history.dropna(subset=["val_perplexity"])
    if df.empty:
        return [], 0.0
    reached = 0
    if "combo_id" in df.columns:
        groups = list(df.groupby("combo_id"))
        for _, group in groups:
            group = group.sort_values(step_col)
            steps = group.loc[group["val_perplexity"] <= target, step_col]
            if steps.empty:
                eval_interval = infer_eval_interval(group[step_col].to_numpy())
                step_values.append(float(group[step_col].max() + eval_interval))
            else:
                step_values.append(float(steps.min()))
                reached += 1
        fraction = reached / max(1, len(groups))
        return step_values, float(fraction)

    group = df.sort_values(step_col)
    steps = group.loc[group["val_perplexity"] <= target, step_col]
    if steps.empty:
        eval_interval = infer_eval_interval(group[step_col].to_numpy())
        return [float(group[step_col].max() + eval_interval)], 0.0
    return [float(steps.min())], 1.0


def aggregate_metrics(run_data: Dict[str, Dict[str, Any]], primary_metric: str) -> Dict[str, Any]:
    metrics: Dict[str, Dict[str, Any]] = {}
    for run_id, data in run_data.items():
        summary = data["summary"]
        run_metrics = data["run_metrics"]
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                metrics.setdefault(key, {})[run_id] = float(value)
        for key, value in run_metrics.items():
            if value is None:
                continue
            metrics.setdefault(key, {})[run_id] = float(value)
    return {
        "primary_metric": primary_metric,
        "metrics": metrics,
    }


def metric_should_minimize(metric_name: str) -> bool:
    name = metric_name.lower()
    return any(key in name for key in ["loss", "perplexity", "error"])


def plot_comparison_bar(metric_values: Dict[str, float], metric_name: str, out_dir: str) -> str:
    plt.figure(figsize=(8, 5))
    run_ids = list(metric_values.keys())
    values = [metric_values[r] for r in run_ids]
    sns.barplot(x=run_ids, y=values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(metric_name)
    plt.title(f"Comparison: {metric_name}")
    for idx, value in enumerate(values):
        plt.text(idx, value, f"{value:.2f}", ha="center", va="bottom")
    plt.tight_layout()
    fig_path = os.path.join(out_dir, f"comparison_{metric_name}_bar_chart.pdf")
    plt.savefig(fig_path)
    plt.close()
    return fig_path


def plot_comparison_boxplot(distributions: Dict[str, List[float]], metric_name: str, out_dir: str) -> str:
    rows = []
    for run_id, values in distributions.items():
        for value in values:
            rows.append({"run_id": run_id, metric_name: value})
    df = pd.DataFrame(rows)
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="run_id", y=metric_name)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Distribution of {metric_name}")
    plt.tight_layout()
    fig_path = os.path.join(out_dir, f"comparison_{metric_name}_boxplot.pdf")
    plt.savefig(fig_path)
    plt.close()
    return fig_path


def plot_steps_to_target(step_metrics: Dict[str, List[float]], out_dir: str) -> str:
    plt.figure(figsize=(8, 5))
    run_ids = list(step_metrics.keys())
    values = [np.median(step_metrics[r]) if step_metrics[r] else np.nan for r in run_ids]
    sns.barplot(x=run_ids, y=values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("median_steps_to_target")
    plt.title("Steps to Target Perplexity")
    for idx, value in enumerate(values):
        if not np.isnan(value):
            plt.text(idx, value, f"{value:.0f}", ha="center", va="bottom")
    plt.tight_layout()
    fig_path = os.path.join(out_dir, "comparison_steps_to_target_bar_chart.pdf")
    plt.savefig(fig_path)
    plt.close()
    return fig_path


def plot_fraction_reached(fractions: Dict[str, float], out_dir: str) -> str:
    plt.figure(figsize=(8, 5))
    run_ids = list(fractions.keys())
    values = [fractions[r] for r in run_ids]
    sns.barplot(x=run_ids, y=values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("fraction_reached_target")
    plt.ylim(0.0, 1.05)
    plt.title("Fraction Reaching Target Perplexity")
    for idx, value in enumerate(values):
        plt.text(idx, value, f"{value:.2f}", ha="center", va="bottom")
    plt.tight_layout()
    fig_path = os.path.join(out_dir, "comparison_fraction_reached_target_bar_chart.pdf")
    plt.savefig(fig_path)
    plt.close()
    return fig_path


def plot_survival_curve(step_metrics: Dict[str, List[float]], out_dir: str) -> str:
    plt.figure(figsize=(8, 5))
    max_step = 0.0
    for values in step_metrics.values():
        if values:
            max_step = max(max_step, max(values))
    grid = np.linspace(0, max_step, num=20) if max_step > 0 else np.array([0])
    for run_id, values in step_metrics.items():
        if not values:
            continue
        steps = np.array(values)
        survival = [(steps > t).mean() for t in grid]
        plt.plot(grid, survival, label=run_id)
    plt.xlabel("step")
    plt.ylabel("fraction_not_reached")
    plt.title("Survival Curve: Steps to Target")
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(out_dir, "comparison_steps_to_target_survival_curve.pdf")
    plt.savefig(fig_path)
    plt.close()
    return fig_path


def plot_metrics_table(metrics: Dict[str, Dict[str, float]], out_dir: str) -> str:
    table_metrics = [
        "perplexity",
        "mean_effective_rank",
        "median_steps_to_target",
        "fraction_reached_target",
        "best_val_perplexity",
    ]
    rows = []
    run_ids = set()
    for metric in table_metrics:
        for run_id, value in metrics.get(metric, {}).items():
            run_ids.add(run_id)
    run_ids = sorted(run_ids)
    for run_id in run_ids:
        row = {"run_id": run_id}
        for metric in table_metrics:
            row[metric] = metrics.get(metric, {}).get(run_id, np.nan)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("run_id")
    fig, ax = plt.subplots(figsize=(min(12, 2 + len(df.columns)), 0.6 + 0.4 * len(df)))
    ax.axis("off")
    table = ax.table(
        cellText=df.round(3).values,
        rowLabels=df.index.tolist(),
        colLabels=df.columns.tolist(),
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    fig.tight_layout()
    fig_path = os.path.join(out_dir, "comparison_metrics_table.pdf")
    fig.savefig(fig_path)
    plt.close(fig)
    return fig_path


def main():
    args = parse_cli_args(sys.argv[1:])

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    wandb_cfg = load_wandb_config(root_dir)
    entity = wandb_cfg["entity"]
    project = wandb_cfg["project"]

    api = wandb.Api()
    run_ids = json.loads(args["run_ids"])
    results_dir = os.path.abspath(args["results_dir"])
    os.makedirs(results_dir, exist_ok=True)

    run_data: Dict[str, Dict[str, Any]] = {}
    generated_files: List[str] = []

    for run_id in run_ids:
        run = api.run(f"{entity}/{project}/{run_id}")
        history = fetch_history(run)
        summary = run.summary._json_dict
        config = dict(run.config)
        ensure_not_trial_run(config, run_id)

        run_data[run_id] = {
            "history": history,
            "summary": summary,
            "config": config,
        }

    baseline_final = []
    for run_id, data in run_data.items():
        history = data["history"]
        final_perplexities = metric_at_step_by_combo(history, "val_perplexity", TARGET_STEP)
        if not final_perplexities:
            final_perplexities = final_metric_by_combo(history, "val_perplexity")
        data["final_perplexities"] = final_perplexities
        if "comparative" in run_id or "baseline" in run_id:
            baseline_final.extend(final_perplexities)

    target = None
    if baseline_final:
        target = 1.01 * float(np.median(baseline_final))

    for run_id, data in run_data.items():
        history = data["history"]
        summary = data["summary"]
        config = data["config"]

        out_dir = os.path.join(results_dir, run_id)
        os.makedirs(out_dir, exist_ok=True)

        final_perplexities = data["final_perplexities"]
        final_perplexity_median = median_or_none(final_perplexities)
        if final_perplexity_median is None:
            final_perplexity_median = summary.get("perplexity") or summary.get("final_val_perplexity")

        best_val_perplexity = None
        if "val_perplexity" in history:
            best_val_perplexity = float(history["val_perplexity"].min())

        r_eff_metric = "val_mean_r_eff" if "val_mean_r_eff" in history else "mean_r_eff"
        final_r_eff = metric_at_step_by_combo(history, r_eff_metric, TARGET_STEP)
        if not final_r_eff:
            final_r_eff = final_metric_by_combo(history, r_eff_metric)
        final_r_eff_median = median_or_none(final_r_eff)

        steps_to_target = []
        fraction_reached = None
        if target is not None:
            steps_to_target, fraction_reached = compute_steps_to_target(history, target)

        run_metrics = {
            "perplexity": final_perplexity_median,
            "final_val_perplexity": final_perplexity_median,
            "best_val_perplexity": best_val_perplexity,
            "mean_effective_rank": final_r_eff_median,
            "median_steps_to_target": float(np.median(steps_to_target)) if steps_to_target else None,
            "fraction_reached_target": fraction_reached,
        }

        data["run_metrics"] = run_metrics
        data["steps_to_target"] = steps_to_target
        data["fraction_reached_target"] = fraction_reached

        metrics_payload = {
            "history": to_jsonable(history.to_dict(orient="list")),
            "summary": to_jsonable(summary),
            "config": to_jsonable(config),
            "derived": {
                "final_perplexities": final_perplexities,
                "median_final_perplexity": final_perplexity_median,
                "final_mean_effective_rank": final_r_eff_median,
                "best_val_perplexity": best_val_perplexity,
                "steps_to_target": steps_to_target,
                "fraction_reached_target": fraction_reached,
                "target_perplexity": target,
            },
        }
        metrics_path = os.path.join(out_dir, "metrics.json")
        save_json(metrics_path, metrics_payload)
        generated_files.append(metrics_path)

        generated_files += plot_learning_curves(history, run_id, out_dir)
        generated_files += plot_rank_lambda(history, run_id, out_dir)
        generated_files += plot_scatter(history, run_id, out_dir)
        generated_files += plot_rank_histogram(history, run_id, out_dir)

    comparison_dir = os.path.join(results_dir, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)

    primary_metric = "perplexity"
    aggregated = aggregate_metrics(run_data, primary_metric)
    aggregated["target_perplexity"] = target

    metrics = aggregated["metrics"]
    primary_values = metrics.get("perplexity", {})

    minimize_metric = metric_should_minimize(primary_metric)
    proposed = {rid: val for rid, val in primary_values.items() if "proposed" in rid}
    baseline = {
        rid: val
        for rid, val in primary_values.items()
        if "comparative" in rid or "baseline" in rid
    }

    if minimize_metric:
        best_proposed = min(proposed.items(), key=lambda x: x[1]) if proposed else (None, None)
        best_baseline = min(baseline.items(), key=lambda x: x[1]) if baseline else (None, None)
    else:
        best_proposed = max(proposed.items(), key=lambda x: x[1]) if proposed else (None, None)
        best_baseline = max(baseline.items(), key=lambda x: x[1]) if baseline else (None, None)

    gap = None
    if best_proposed[1] is not None and best_baseline[1] is not None:
        gap = (best_proposed[1] - best_baseline[1]) / best_baseline[1] * 100.0
        if minimize_metric:
            gap = -gap

    aggregated["best_proposed"] = {"run_id": best_proposed[0], "value": best_proposed[1]}
    aggregated["best_baseline"] = {"run_id": best_baseline[0], "value": best_baseline[1]}
    aggregated["gap"] = gap

    step_metrics = {run_id: data.get("steps_to_target", []) for run_id, data in run_data.items()}
    fractions = {
        run_id: data.get("fraction_reached_target")
        for run_id, data in run_data.items()
        if data.get("fraction_reached_target") is not None
    }

    if step_metrics:
        aggregated["metrics"]["median_steps_to_target"] = {
            run_id: float(np.median(vals)) if vals else None for run_id, vals in step_metrics.items()
        }
    if fractions:
        aggregated["metrics"]["fraction_reached_target"] = fractions

    stats_results = {}
    if proposed and baseline:
        proposed_vals = list(proposed.values())
        baseline_vals = list(baseline.values())
        if len(proposed_vals) > 1 and len(baseline_vals) > 1:
            t_stat, p_val = stats.ttest_ind(proposed_vals, baseline_vals, equal_var=False)
            stats_results = {"t_stat": float(t_stat), "p_value": float(p_val)}
    aggregated["stats"] = stats_results

    aggregated_path = os.path.join(comparison_dir, "aggregated_metrics.json")
    save_json(aggregated_path, to_jsonable(aggregated))
    generated_files.append(aggregated_path)

    if primary_values:
        generated_files.append(plot_comparison_bar(primary_values, "perplexity", comparison_dir))
    if "mean_effective_rank" in metrics:
        generated_files.append(plot_comparison_bar(metrics["mean_effective_rank"], "mean_effective_rank", comparison_dir))
    if step_metrics:
        generated_files.append(plot_steps_to_target(step_metrics, comparison_dir))
        generated_files.append(plot_survival_curve(step_metrics, comparison_dir))
    if fractions:
        generated_files.append(plot_fraction_reached(fractions, comparison_dir))

    distribution_values = {
        run_id: data.get("final_perplexities", []) for run_id, data in run_data.items()
    }
    if any(values for values in distribution_values.values()):
        generated_files.append(plot_comparison_boxplot(distribution_values, "final_perplexity", comparison_dir))

    generated_files.append(plot_metrics_table(metrics, comparison_dir))

    for path in generated_files:
        print(path)


if __name__ == "__main__":
    main()
