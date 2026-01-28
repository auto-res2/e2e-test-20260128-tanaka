import warnings
from typing import Dict, Optional, Tuple

import torch
from datasets import DatasetDict, load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


def load_raw_dataset(dataset_cfg, cache_dir: str) -> DatasetDict:
    name = str(dataset_cfg.name).lower()
    if name in {"wikitext-2", "wikitext2", "wikitext"}:
        raw = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir=cache_dir)
    elif name in {"ptb_text_only", "ptb"}:
        raw = load_dataset("FALcon6/ptb_text_only", "penn_treebank", cache_dir=cache_dir)
    else:
        raw = load_dataset(dataset_cfg.name, cache_dir=cache_dir)
    return raw


def ensure_splits(raw: DatasetDict, split_cfg: Dict[str, str], seed: int) -> DatasetDict:
    train_split = split_cfg.get("train", "train")
    val_split = split_cfg.get("validation", "validation")
    test_split = split_cfg.get("test", "test")

    datasets = DatasetDict()
    if train_split not in raw:
        raise ValueError(f"Train split '{train_split}' not found in dataset splits: {list(raw.keys())}")
    datasets["train"] = raw[train_split]

    if val_split in raw:
        datasets["validation"] = raw[val_split]
    else:
        split = datasets["train"].train_test_split(test_size=0.05, seed=seed)
        datasets["train"] = split["train"]
        datasets["validation"] = split["test"]

    if test_split in raw:
        datasets["test"] = raw[test_split]
    else:
        datasets["test"] = datasets["validation"]

    return datasets


def resolve_text_column(datasets: DatasetDict, requested: Optional[str]) -> str:
    columns = datasets["train"].column_names
    if requested and requested in columns:
        return requested
    if "text" in columns:
        warnings.warn(f"Requested text column '{requested}' not found; using 'text' instead.")
        return "text"
    warnings.warn(
        f"Requested text column '{requested}' not found; using '{columns[0]}' instead."
    )
    return columns[0]


def load_tokenized_datasets(cfg, tokenizer, cache_dir: str) -> DatasetDict:
    raw = load_raw_dataset(cfg.dataset, cache_dir=cache_dir)
    datasets = ensure_splits(raw, cfg.dataset.split, seed=int(cfg.training.seed))

    text_column = resolve_text_column(datasets, cfg.dataset.text_column)

    def non_empty(example):
        text = example.get(text_column, "")
        return text is not None and len(str(text).strip()) > 0

    datasets = datasets.filter(non_empty)

    truncation = not bool(cfg.dataset.pack_to_max_length)
    max_length = int(cfg.dataset.max_length or cfg.training.seq_len)

    def tokenize_fn(examples):
        return tokenizer(
            examples[text_column],
            return_attention_mask=False,
            truncation=truncation,
            max_length=max_length if truncation else None,
        )

    tokenized = datasets.map(
        tokenize_fn,
        batched=True,
        remove_columns=datasets["train"].column_names,
        desc="Tokenizing",
    )

    if cfg.dataset.pack_to_max_length:
        block_size = int(cfg.dataset.max_length or cfg.training.seq_len)

        def group_texts(examples):
            concatenated = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated["input_ids"])
            if total_length >= block_size:
                total_length = (total_length // block_size) * block_size
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated.items()
            }
            return result

        tokenized = tokenized.map(group_texts, batched=True, desc="Packing")

    return tokenized


class CausalLMCollator:
    def __init__(self, tokenizer, max_length: Optional[int] = None):
        if tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer is missing pad_token_id.")
        self.pad_token_id = tokenizer.pad_token_id
        self.max_length = max_length

    def __call__(self, batch):
        input_ids = [torch.tensor(example["input_ids"], dtype=torch.long) for example in batch]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        if self.max_length is not None:
            input_ids = input_ids[:, : self.max_length]
        attention_mask = (input_ids != self.pad_token_id).long()
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def create_dataloaders(
    tokenized_datasets: DatasetDict,
    tokenizer,
    cfg,
    seed: int,
    limit_train_batches: Optional[int] = None,
    limit_eval_batches: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    batch_size = int(cfg.training.batch_size)
    max_length = int(cfg.dataset.max_length or cfg.training.seq_len)
    collator = CausalLMCollator(tokenizer, max_length=max_length)

    train_dataset = tokenized_datasets["train"]
    val_dataset = tokenized_datasets["validation"]
    test_dataset = tokenized_datasets["test"]

    if limit_train_batches is not None:
        max_samples = min(len(train_dataset), limit_train_batches * batch_size)
        train_dataset = train_dataset.select(range(max_samples))
    if limit_eval_batches is not None:
        max_samples = min(len(val_dataset), limit_eval_batches * batch_size)
        val_dataset = val_dataset.select(range(max_samples))

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        generator=generator,
        drop_last=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, test_loader
