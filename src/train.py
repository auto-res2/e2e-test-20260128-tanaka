import copy
import gc
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import hydra
import numpy as np
import optuna
import torch
import torch.nn.functional as F
import wandb
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from .model import (
    AutoCoreVolLoRAController,
    autocorevollora_stats,
    build_lora_model,
    collect_lora_pairs,
    load_base_model,
    load_tokenizer,
    ortholora_penalty,
    precision_to_dtype,
    set_seed,
    vollora_factorwise_logdet,
)
from .preprocess import create_dataloaders, load_tokenized_datasets


def apply_mode_overrides(cfg):
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        if getattr(cfg, "optuna", None) is not None and "n_trials" in cfg.optuna:
            cfg.optuna.n_trials = 0
        if getattr(cfg, "run", None) is not None:
            if getattr(cfg.run, "optuna", None) is not None and "n_trials" in cfg.run.optuna:
                cfg.run.optuna.n_trials = 0
            if getattr(cfg.run, "training", None) is not None:
                training = cfg.run.training
                if training.max_steps is not None:
                    training.max_steps = min(int(training.max_steps), 2)
                else:
                    training.epochs = 1
                training.eval_every_steps = 1
                if training.get("robustness") is not None:
                    training.robustness.seeds = [int(training.seed)]
                    training.robustness.lr_multipliers = [1.0]
    elif cfg.mode == "full":
        cfg.wandb.mode = "online"
    return cfg


def ensure_list(value):
    # Convert OmegaConf types to plain Python types
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def prepare_cache_dirs(cache_dir: str) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    os.environ.setdefault("HF_HOME", cache_dir)
    os.environ.setdefault("TRANSFORMERS_CACHE", cache_dir)
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(cache_dir, "datasets"))


def infer_use_qlora(cfg) -> bool:
    if bool(cfg.model.get("use_qlora", False)):
        return True
    use_for = cfg.training.get("use_qlora_for", []) if cfg.training is not None else []
    return cfg.model.name in use_for


def resolve_method_family(method_name: str) -> str:
    if method_name is None:
        return "lora"
    name = str(method_name).lower()
    if "autocore" in name:
        return "autocore"
    if "ortho" in name:
        return "ortho"
    if "vollora" in name:
        return "vollora"
    return "lora"


def compute_grad_norm(parameters: List[torch.nn.Parameter]) -> float:
    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        param_norm = param.grad.detach().data.norm(2).item()
        total += param_norm**2
    return math.sqrt(total)


def assert_gradients(parameters: List[torch.nn.Parameter]) -> float:
    grads = [param.grad for param in parameters if param.requires_grad]
    assert all(grad is not None for grad in grads), "Missing gradients before optimizer step."
    grad_norm = compute_grad_norm(parameters)
    assert grad_norm > 0.0 and math.isfinite(grad_norm), "Gradients are zero or non-finite."
    return grad_norm


def compute_aux_grad_norm(logdet: torch.Tensor, parameters: List[torch.nn.Parameter]) -> float:
    grads = torch.autograd.grad(
        logdet,
        parameters,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    total = 0.0
    for grad in grads:
        if grad is None:
            continue
        total += grad.detach().data.norm(2).item() ** 2
    return math.sqrt(total)


def compute_causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    if attention_mask is not None:
        shift_mask = attention_mask[:, 1:].contiguous()
        shift_labels = shift_labels.masked_fill(shift_mask == 0, -100)
    vocab_size = shift_logits.size(-1)
    loss = F.cross_entropy(shift_logits.view(-1, vocab_size), shift_labels.view(-1), ignore_index=-100)
    return loss


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    precision: str,
    rho: float,
    eps: float,
    max_batches: Optional[int] = None,
) -> Tuple[float, float, float, List[float]]:
    model.eval()
    losses = []
    autocast_dtype = precision_to_dtype(precision)
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_dtype != torch.float32,
            ):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = compute_causal_lm_loss(logits, labels, attention_mask)
            losses.append(loss.detach().float().item())
    mean_loss = float(sum(losses) / max(1, len(losses)))
    perplexity = float(math.exp(mean_loss))
    with torch.no_grad():
        _, mean_r_eff, r_eff_list = autocorevollora_stats(
            model,
            rho=rho,
            eps=eps,
            return_per_module=True,
        )
    mean_r_eff_value = float(mean_r_eff.detach().cpu().item())
    r_eff_values = [float(val.detach().cpu().item()) for val in r_eff_list]
    return mean_loss, perplexity, mean_r_eff_value, r_eff_values


def apply_hyperparams(cfg, hyperparams: Dict[str, Any]):
    cfg = copy.deepcopy(cfg)
    for name, value in hyperparams.items():
        OmegaConf.update(cfg, name, value, merge=True)
    return cfg


def resolve_param_path(param_name: str) -> str:
    if param_name.startswith("training."):
        return param_name
    if param_name.startswith("autocorevollora."):
        return f"training.{param_name}"
    if param_name.startswith("ortholora."):
        return f"training.{param_name}"
    if param_name.startswith("vollora."):
        return f"training.{param_name}"
    if param_name in {"learning_rate", "lora_dropout", "weight_decay", "gradient_clip"}:
        return f"training.{param_name}"
    return param_name


def run_optuna(
    cfg,
    tokenized_datasets,
    tokenizer,
    device,
    cache_dir: str,
    rank: int,
    gamma_values: List[float],
) -> Dict[str, Any]:
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        trial_params = {}
        for space in cfg.optuna.search_spaces:
            param_name = space["param_name"]
            dtype = space["distribution_type"]
            if dtype == "loguniform":
                value = trial.suggest_float(param_name, float(space["low"]), float(space["high"]), log=True)
            elif dtype == "uniform":
                value = trial.suggest_float(param_name, float(space["low"]), float(space["high"]))
            elif dtype == "categorical":
                value = trial.suggest_categorical(param_name, list(space["choices"]))
            else:
                raise ValueError(f"Unsupported distribution type: {dtype}")
            trial_params[resolve_param_path(param_name)] = value

        if "training.autocorevollora.gamma" not in trial_params and gamma_values:
            trial_params["training.autocorevollora.gamma"] = gamma_values[0]

        trial_cfg = apply_hyperparams(cfg, trial_params)
        result, _ = train_single(
            trial_cfg,
            tokenized_datasets,
            tokenizer,
            device,
            rank=rank,
            seed=int(trial_cfg.training.seed),
            lr_mult=1.0,
            combo_id=f"optuna_r{rank}_trial{trial.number}",
            cache_dir=cache_dir,
            global_step_offset=0,
            log_to_wandb=False,
            wandb_run=None,
        )
        return float(result["final_val_perplexity"])

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=int(cfg.training.seed)))
    study.optimize(objective, n_trials=int(cfg.optuna.n_trials))

    best_params = {resolve_param_path(k): v for k, v in study.best_trial.params.items()}
    return best_params


def train_single(
    cfg,
    tokenized_datasets,
    tokenizer,
    device: torch.device,
    rank: int,
    seed: int,
    lr_mult: float,
    combo_id: str,
    cache_dir: str,
    global_step_offset: int,
    log_to_wandb: bool,
    wandb_run: Optional[wandb.sdk.wandb_run.Run],
) -> Tuple[Dict[str, Any], int]:
    set_seed(seed)
    training_cfg = cfg.training
    method_family = resolve_method_family(cfg.method)
    limit_train_batches = 2 if cfg.mode == "trial" else None
    limit_eval_batches = 1 if cfg.mode == "trial" else None

    train_loader, val_loader, _ = create_dataloaders(
        tokenized_datasets,
        tokenizer,
        cfg,
        seed=seed,
        limit_train_batches=limit_train_batches,
        limit_eval_batches=limit_eval_batches,
    )

    use_qlora = infer_use_qlora(cfg)
    model = load_base_model(
        cfg.model.name,
        use_qlora=use_qlora,
        precision=training_cfg.precision,
        cache_dir=cache_dir,
        qlora_4bit=training_cfg.get("qlora_4bit", {}),
        device=device,
    )

    if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    model = build_lora_model(model, cfg, rank)

    assert tokenizer.pad_token_id is not None, "Tokenizer pad_token_id is missing."
    output_embeddings = model.get_output_embeddings()
    assert output_embeddings is not None, "Model output embeddings are missing."
    assert (
        output_embeddings.weight.shape[0] == len(tokenizer)
    ), "Model output dimension does not match tokenizer vocabulary size."

    lora_pairs = collect_lora_pairs(model)
    assert len(lora_pairs) > 0, "No LoRA parameters found after applying adapters."

    max_steps = int(training_cfg.max_steps) if training_cfg.max_steps is not None else None
    grad_accum_steps = int(training_cfg.gradient_accumulation_steps)
    assert grad_accum_steps > 0, "gradient_accumulation_steps must be positive."
    if max_steps is None:
        assert training_cfg.epochs is not None, "Either max_steps or epochs must be set."
        steps_per_epoch = math.ceil(len(train_loader) / grad_accum_steps)
        max_steps = steps_per_epoch * int(training_cfg.epochs)

    eval_every_steps = int(training_cfg.eval_every_steps or max_steps)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(training_cfg.learning_rate) * float(lr_mult),
        weight_decay=float(training_cfg.weight_decay),
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(training_cfg.warmup_steps),
        num_training_steps=max_steps,
    )

    autocore_cfg = training_cfg.get("autocorevollora", {}) or {}
    gamma_value = autocore_cfg.get("gamma", None)
    if isinstance(gamma_value, list):
        gamma_value = gamma_value[0]
    if gamma_value is None:
        gamma_value = 0.75

    controller = None
    if method_family == "autocore":
        controller = AutoCoreVolLoRAController(
            r=rank,
            gamma=float(gamma_value),
            lam_init=float(autocore_cfg.get("lambda_init", 0.0)),
            lam_max=float(autocore_cfg.get("lambda_max", 0.05)),
            eta_lam=float(autocore_cfg.get("eta_lambda", 1e-3)),
            warmup_steps=int(autocore_cfg.get("lambda_warmup_steps", 100)),
        )

    rho = float(autocore_cfg.get("rho", 1e-2))
    eps = float(autocore_cfg.get("eps", 1e-8))

    vollora_cfg = training_cfg.get("vollora", {}) or {}
    vollora_lambda = float(vollora_cfg.get("lambda", vollora_cfg.get("lam", 1e-3)))
    vollora_warmup = int(vollora_cfg.get("lambda_warmup_steps", training_cfg.warmup_steps))

    ortholora_cfg = training_cfg.get("ortholora", {}) or {}
    ortholora_lambda = float(ortholora_cfg.get("lambda", ortholora_cfg.get("lam", 1e-3)))
    ortholora_warmup = int(ortholora_cfg.get("lambda_warmup_steps", training_cfg.warmup_steps))

    precision = str(training_cfg.precision).lower()
    autocast_dtype = precision_to_dtype(precision)
    scaler = torch.cuda.amp.GradScaler(
        enabled=torch.cuda.is_available() and precision in {"fp16", "float16"}
    )

    global_step = global_step_offset
    best_val_perplexity = float("inf")
    final_val_perplexity = float("inf")
    final_val_loss = float("inf")
    final_mean_r_eff = 0.0
    progress = tqdm(range(max_steps), desc=f"{combo_id}", disable=cfg.mode == "trial")
    train_iterator = iter(train_loader)

    for step in progress:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        task_loss_accum = 0.0
        total_loss_accum = 0.0
        autocore_logdet_accum = 0.0
        reg_term_accum = 0.0
        mean_r_eff_accum = 0.0
        aux_grad_norm = None
        lam_value = 0.0
        reg_active = False

        for accum_idx in range(grad_accum_steps):
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                batch = next(train_iterator)

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if step == 0 and accum_idx == 0:
                assert input_ids.shape == labels.shape, "Input/label shape mismatch at step 0."

            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_dtype != torch.float32,
            ):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                task_loss = compute_causal_lm_loss(logits, labels, attention_mask)

                if method_family == "autocore":
                    autocore_logdet, mean_r_eff = autocorevollora_stats(model, rho=rho, eps=eps)
                    autocore_logdet = autocore_logdet.to(task_loss.dtype)
                    if controller is not None:
                        if accum_idx == 0:
                            lam_value = controller.step(step, float(mean_r_eff.detach().cpu().item()))
                        else:
                            lam_value = controller.lam
                    reg_active = controller is not None and step >= controller.warmup_steps
                    total_loss = task_loss - float(lam_value) * autocore_logdet if reg_active else task_loss
                    reg_term = autocore_logdet
                elif method_family == "vollora":
                    reg_term = vollora_factorwise_logdet(model, rho=rho, eps=eps).to(task_loss.dtype)
                    with torch.no_grad():
                        autocore_logdet, mean_r_eff = autocorevollora_stats(model, rho=rho, eps=eps)
                    reg_active = step >= vollora_warmup
                    lam_value = vollora_lambda
                    total_loss = task_loss - float(lam_value) * reg_term if reg_active else task_loss
                elif method_family == "ortho":
                    reg_term = ortholora_penalty(model, eps=eps).to(task_loss.dtype)
                    with torch.no_grad():
                        autocore_logdet, mean_r_eff = autocorevollora_stats(model, rho=rho, eps=eps)
                    reg_active = step >= ortholora_warmup
                    lam_value = ortholora_lambda
                    total_loss = task_loss + float(lam_value) * reg_term if reg_active else task_loss
                else:
                    with torch.no_grad():
                        autocore_logdet, mean_r_eff = autocorevollora_stats(model, rho=rho, eps=eps)
                    reg_term = torch.tensor(0.0, device=task_loss.device, dtype=task_loss.dtype)
                    total_loss = task_loss

                loss = total_loss / grad_accum_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            task_loss_accum += float(task_loss.detach().float().item())
            total_loss_accum += float(total_loss.detach().float().item())
            autocore_logdet_accum += float(autocore_logdet.detach().float().item())
            reg_term_accum += float(reg_term.detach().float().item())
            mean_r_eff_accum += float(mean_r_eff.detach().float().item())

            if (
                method_family in {"autocore", "vollora"}
                and accum_idx == 0
                and step % eval_every_steps == 0
                and reg_term.requires_grad
            ):
                aux_grad_norm = compute_aux_grad_norm(reg_term, trainable_params)

        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        grad_norm = assert_gradients(trainable_params)
        if float(training_cfg.gradient_clip) > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, float(training_cfg.gradient_clip))

        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()

        lr = scheduler.get_last_lr()[0]
        train_loss = task_loss_accum / grad_accum_steps
        train_total_loss = total_loss_accum / grad_accum_steps
        train_perplexity = float(math.exp(train_loss))
        mean_r_eff_value = mean_r_eff_accum / grad_accum_steps
        reg_loss_value = reg_term_accum / grad_accum_steps

        metrics = {
            "train_loss": train_loss,
            "train_total_loss": train_total_loss,
            "train_perplexity": train_perplexity,
            "autocore_logdet": autocore_logdet_accum / grad_accum_steps,
            "reg_loss": reg_loss_value,
            "mean_r_eff": mean_r_eff_value,
            "lambda": float(lam_value) if lam_value is not None else 0.0,
            "reg_active": int(reg_active),
            "grad_norm": grad_norm,
            "lr": lr,
            "rank": int(rank),
            "seed": int(seed),
            "lr_mult": float(lr_mult),
            "combo_id": combo_id,
            "combo_step": int(step + 1),
        }
        if aux_grad_norm is not None:
            metrics["reg_grad_norm"] = aux_grad_norm

        if log_to_wandb and wandb_run is not None:
            wandb.log(metrics, step=global_step)

        if (step + 1) % eval_every_steps == 0 or step == max_steps - 1:
            val_loss, val_perplexity, val_mean_r_eff, r_eff_list = evaluate_model(
                model,
                val_loader,
                device,
                precision=precision,
                rho=rho,
                eps=eps,
                max_batches=limit_eval_batches,
            )
            final_val_perplexity = val_perplexity
            final_val_loss = val_loss
            final_mean_r_eff = val_mean_r_eff
            if val_perplexity < best_val_perplexity:
                best_val_perplexity = val_perplexity

            eval_metrics = {
                "val_loss": val_loss,
                "val_perplexity": val_perplexity,
                "val_mean_r_eff": val_mean_r_eff,
                "best_val_perplexity": best_val_perplexity,
                "rank": int(rank),
                "seed": int(seed),
                "lr_mult": float(lr_mult),
                "combo_id": combo_id,
                "combo_step": int(step + 1),
            }
            if log_to_wandb and wandb_run is not None:
                wandb.log(eval_metrics, step=global_step)
                if r_eff_list:
                    wandb.log(
                        {"r_eff_hist": wandb.Histogram(r_eff_list)},
                        step=global_step,
                    )

        global_step += 1

    result = {
        "combo_id": combo_id,
        "rank": int(rank),
        "seed": int(seed),
        "lr_mult": float(lr_mult),
        "best_val_perplexity": float(best_val_perplexity),
        "final_val_perplexity": float(final_val_perplexity),
        "final_val_loss": float(final_val_loss),
        "final_mean_r_eff": float(final_mean_r_eff),
    }

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result, global_step


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg):
    cfg = apply_mode_overrides(cfg)
    # Temporarily disable struct mode to allow merging run config with new keys
    OmegaConf.set_struct(cfg, False)
    cfg = OmegaConf.merge(cfg, cfg.run)
    OmegaConf.set_struct(cfg, True)
    if not hasattr(cfg, "run") or cfg.run is None:
        raise ValueError("Run configuration must be provided via run=<run_id>.")
    if not hasattr(cfg, "run_id"):
        cfg.run_id = cfg.run.run_id

    root = get_original_cwd()
    results_dir = cfg.results_dir
    if not os.path.isabs(results_dir):
        results_dir = os.path.abspath(os.path.join(root, results_dir))
    os.makedirs(results_dir, exist_ok=True)

    cache_dir = os.path.join(root, ".cache")
    prepare_cache_dirs(cache_dir)

    assert cfg.run.run_id, "Run configuration must include run_id."

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(cfg.model.name, cache_dir)
    tokenized_datasets = load_tokenized_datasets(cfg, tokenizer, cache_dir)

    ranks = ensure_list(cfg.training.lora_rank)
    seeds = ensure_list(cfg.training.robustness.seeds) if cfg.training.get("robustness") else [int(cfg.training.seed)]
    lr_mults = (
        ensure_list(cfg.training.robustness.lr_multipliers) if cfg.training.get("robustness") else [1.0]
    )
    gamma_values = []
    if resolve_method_family(cfg.method) == "autocore" and cfg.training.get("autocorevollora") is not None:
        gamma_values = ensure_list(cfg.training.autocorevollora.gamma)

    best_params_by_rank: Dict[int, Dict[str, Any]] = {}
    if cfg.optuna is not None and int(cfg.optuna.n_trials) > 0:
        for rank in ranks:
            best_params_by_rank[rank] = run_optuna(
                cfg,
                tokenized_datasets,
                tokenizer,
                device,
                cache_dir,
                rank=rank,
                gamma_values=gamma_values,
            )

    log_to_wandb = cfg.wandb.mode != "disabled"
    wandb_run = None
    if log_to_wandb:
        wandb_config = OmegaConf.to_container(cfg, resolve=True)
        wandb_run = wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            id=cfg.run.run_id,
            config=wandb_config,
            resume="allow",
        )
        print(f"WandB URL: {wandb_run.url}")

    all_results = []
    global_step_offset = 0

    for rank in ranks:
        best_params = best_params_by_rank.get(rank, {})
        gamma_loop = gamma_values if (resolve_method_family(cfg.method) == "autocore" and not best_params) else [None]

        for gamma in gamma_loop:
            hyperparams = dict(best_params)
            if gamma is not None:
                hyperparams["training.autocorevollora.gamma"] = float(gamma)

            cfg_run = apply_hyperparams(cfg, hyperparams) if hyperparams else cfg

            for seed in seeds:
                for lr_mult in lr_mults:
                    combo_suffix = f"r{rank}_s{seed}_lr{lr_mult}"
                    if gamma is not None:
                        combo_suffix += f"_g{gamma}"
                    combo_id = f"{cfg.run.run_id}_{combo_suffix}"

                    result, global_step_offset = train_single(
                        cfg_run,
                        tokenized_datasets,
                        tokenizer,
                        device,
                        rank=rank,
                        seed=int(seed),
                        lr_mult=float(lr_mult),
                        combo_id=combo_id,
                        cache_dir=cache_dir,
                        global_step_offset=global_step_offset,
                        log_to_wandb=log_to_wandb,
                        wandb_run=wandb_run,
                    )
                    all_results.append(result)

    if wandb_run is not None:
        best_vals = [res["best_val_perplexity"] for res in all_results]
        final_vals = [res["final_val_perplexity"] for res in all_results]
        mean_r_eff = [res["final_mean_r_eff"] for res in all_results]
        if final_vals:
            wandb_run.summary["perplexity"] = float(np.median(final_vals))
            wandb_run.summary["final_val_perplexity_median"] = float(np.median(final_vals))
            wandb_run.summary["final_val_perplexity_mean"] = float(np.mean(final_vals))
        if best_vals:
            wandb_run.summary["best_val_perplexity"] = float(min(best_vals))
        if mean_r_eff:
            wandb_run.summary["mean_final_r_eff"] = float(np.mean(mean_r_eff))
        wandb_run.summary["num_combos"] = len(all_results)
        wandb_run.summary["combo_results"] = json.dumps(all_results)
        wandb_run.finish()


if __name__ == "__main__":
    main()
