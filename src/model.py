import random
from typing import List, Tuple

import numpy as np
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from peft.tuners.lora import LoraLayer
except Exception:  # pragma: no cover
    from peft.tuners.lora.layer import LoraLayer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def precision_to_dtype(precision: str) -> torch.dtype:
    precision = (precision or "fp32").lower()
    if precision in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if precision in {"fp16", "float16"}:
        return torch.float16
    return torch.float32


def parse_lora_alpha(lora_alpha, r: int) -> float:
    if isinstance(lora_alpha, (int, float)):
        return float(lora_alpha)
    if isinstance(lora_alpha, str):
        expr = lora_alpha.replace(" ", "")
        try:
            value = eval(expr, {"__builtins__": {}}, {"r": r})
        except Exception:
            value = float(expr)
        return float(value)
    raise ValueError(f"Unsupported lora_alpha type: {type(lora_alpha)}")


def resolve_target_modules(model, target_modules: List[str]) -> List[str]:
    module_names = [name for name, _ in model.named_modules()]
    if not target_modules:
        raise ValueError("target_modules must be provided for LoRA adapters.")
    resolved = [t for t in target_modules if any(name.endswith(t) for name in module_names)]
    if resolved:
        return resolved
    fallback = ["c_attn", "qkv_proj", "attn.c_attn"]
    for fb in fallback:
        if any(name.endswith(fb) for name in module_names):
            return [fb]
    raise ValueError(
        f"None of the target modules {target_modules} were found in the model. "
        f"Available modules include: {module_names[:50]}..."
    )


def load_tokenizer(model_name: str, cache_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        else:
            tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_base_model(
    model_name: str,
    use_qlora: bool,
    precision: str,
    cache_dir: str,
    qlora_4bit: dict,
    device: torch.device,
):
    torch_dtype = precision_to_dtype(precision)
    if use_qlora:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=bool(qlora_4bit.get("load_in_4bit", True)),
            bnb_4bit_quant_type=str(qlora_4bit.get("bnb_4bit_quant_type", "nf4")),
            bnb_4bit_use_double_quant=bool(qlora_4bit.get("bnb_4bit_use_double_quant", True)),
            bnb_4bit_compute_dtype=precision_to_dtype(
                qlora_4bit.get("bnb_4bit_compute_dtype", "bf16")
            ),
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            device_map="auto",
            quantization_config=quant_config,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        model.to(device)
    model.config.use_cache = False
    return model


def build_lora_model(model, cfg, rank: int):
    lora_alpha = parse_lora_alpha(cfg.training.lora_alpha, rank)
    target_modules = resolve_target_modules(model, cfg.training.target_modules)
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        lora_dropout=float(cfg.training.lora_dropout),
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    return model


def collect_lora_pairs(model) -> List[Tuple[str, torch.Tensor, torch.Tensor]]:
    pairs = []
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            for adapter_name in module.lora_A.keys():
                A = module.lora_A[adapter_name].weight
                B = module.lora_B[adapter_name].weight
                pairs.append((f"{name}.{adapter_name}", A, B))
    return pairs


def ortholora_penalty(model, eps: float = 1e-8) -> torch.Tensor:
    pairs = collect_lora_pairs(model)
    if len(pairs) == 0:
        raise RuntimeError("No LoRA modules found for Ortho-LoRA penalty.")
    device = pairs[0][1].device
    total = torch.tensor(0.0, device=device, dtype=torch.float32)
    for _, A, B in pairs:
        r = A.shape[0]
        A_f = A.float()
        B_f = B.float()
        I = torch.eye(r, device=device, dtype=torch.float32)
        AA = A_f @ A_f.T
        BB = B_f.T @ B_f
        total = total + torch.norm(AA - I, p="fro") ** 2 + torch.norm(BB - I, p="fro") ** 2
    return total


def vollora_factorwise_logdet(model, rho: float = 1e-2, eps: float = 1e-8) -> torch.Tensor:
    pairs = collect_lora_pairs(model)
    if len(pairs) == 0:
        raise RuntimeError("No LoRA modules found for VolLoRA-factorwise penalty.")
    device = pairs[0][1].device
    total = torch.tensor(0.0, device=device, dtype=torch.float32)
    for _, A, B in pairs:
        r = A.shape[0]
        A_f = A.float()
        B_f = B.float()
        I = torch.eye(r, device=device, dtype=torch.float32)
        GA = A_f @ A_f.T
        GB = B_f.T @ B_f
        GA = GA / (torch.trace(GA) + eps)
        GB = GB / (torch.trace(GB) + eps)
        sign_a, logdet_a = torch.linalg.slogdet(GA + rho * I)
        sign_b, logdet_b = torch.linalg.slogdet(GB + rho * I)
        if sign_a > 0:
            total = total + logdet_a
        if sign_b > 0:
            total = total + logdet_b
    return total


def autocorevollora_stats(
    model,
    rho: float = 1e-2,
    eps: float = 1e-8,
    return_per_module: bool = False,
):
    pairs = collect_lora_pairs(model)
    if len(pairs) == 0:
        raise RuntimeError("No LoRA modules found for AutoCoreVolLoRA statistics.")
    device = pairs[0][1].device
    total_logdet = torch.tensor(0.0, device=device, dtype=torch.float32)
    eff_ranks = []
    for _, A, B in pairs:
        r = A.shape[0]
        A_f = A.float()
        B_f = B.float()
        I = torch.eye(r, device=device, dtype=torch.float32)
        GB = B_f.T @ B_f
        GA = A_f @ A_f.T

        evalsB, evecsB = torch.linalg.eigh(GB + eps * I)
        evalsB = torch.clamp(evalsB, min=0.0)
        sqrtGB = (evecsB * torch.sqrt(evalsB)).matmul(evecsB.T)

        C = sqrtGB @ GA @ sqrtGB
        trC = torch.trace(C)
        Cn = C / (trC + eps)

        sign, logdet = torch.linalg.slogdet(Cn + rho * I)
        if sign > 0:
            total_logdet = total_logdet + logdet

        evalsC = torch.linalg.eigvalsh(Cn + eps * I)
        p = torch.clamp(evalsC, min=eps)
        p = p / p.sum()
        r_eff = torch.exp(-(p * torch.log(p)).sum())
        eff_ranks.append(r_eff)

    mean_r_eff = torch.stack(eff_ranks).mean()
    if return_per_module:
        return total_logdet, mean_r_eff, eff_ranks
    return total_logdet, mean_r_eff


class AutoCoreVolLoRAController:
    def __init__(
        self,
        r: int,
        gamma: float = 0.75,
        lam_init: float = 0.0,
        lam_max: float = 0.05,
        eta_lam: float = 1e-3,
        warmup_steps: int = 100,
    ):
        self.r = r
        self.gamma = gamma
        self.lam = lam_init
        self.lam_max = lam_max
        self.eta_lam = eta_lam
        self.warmup_steps = warmup_steps

    def step(self, step_idx: int, mean_eff_rank_detached: float) -> float:
        if step_idx < self.warmup_steps:
            return self.lam
        target = self.gamma * self.r
        self.lam = float(self.lam + self.eta_lam * (target - mean_eff_rank_detached))
        self.lam = max(0.0, min(self.lam, self.lam_max))
        return self.lam
