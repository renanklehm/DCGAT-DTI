from __future__ import annotations

import argparse
import contextlib
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any

from scripts.common import (
    REPO_ROOT,
    compose_cfg,
    default_best_param_name,
    ensure_repo_on_path,
    normalize_dir,
    normalize_split_strategy,
    pick_checkpoint,
    sanitize_slug,
    update_best_params,
)
from scripts.custom_dataset_utils import (
    build_custom_tables,
    build_prediction_dataset,
    custom_serializer_names,
    export_checkpoint_to_safetensors,
    generate_custom_embeddings,
    predict_checkpoint_on_dataset,
    read_custom_triplets,
    save_custom_tables,
    save_exclusions_report,
    save_prediction_export,
)


DEFAULT_SERIALIZED_DIR = REPO_ROOT / "datasets" / "serialized"
DEFAULT_ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "custom_training"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train DCGAT-DTI on a custom smiles|sequence|activation dataset, evaluate on a held-out split, "
            "and export predictions plus safetensors weights."
        )
    )
    parser.add_argument("--custom-data", type=Path, required=True, help="Path to the custom dataset file.")
    parser.add_argument("--delimiter", default="|", help="Field delimiter for CSV input. Defaults to '|'.")
    parser.add_argument("--has-header", action="store_true", help="Treat the first CSV row as a header row.")
    parser.add_argument(
        "--base-config",
        default="drugbank_train_GAT.yaml",
        choices=(
            "drugbank_train_GAT.yaml",
            "bindingDB_train_GAT.yaml",
            "yamanishi_train.yaml",
            "luo_train.yaml",
        ),
        help="Existing Hydra config used as the model/template baseline.",
    )
    parser.add_argument(
        "--best-param-name",
        default=None,
        help="Best-params file from configs/best_params applied before training. Defaults to one matching the split mode.",
    )
    parser.add_argument(
        "--split-strategy",
        choices=("warm", "random", "cold_drug", "cold_target", "cold_full"),
        default="warm",
        help="How to split the custom dataset into train/val/test.",
    )
    parser.add_argument(
        "--balanced",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use balanced splits like the paper scenarios. Disable for unbalanced runs.",
    )
    parser.add_argument(
        "--unbalanced-ratio",
        type=int,
        default=10,
        help="Negative:positive ratio when --no-balanced is used and downsampling is desired.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.79, help="Training ratio before balancing.")
    parser.add_argument("--val-ratio", type=float, default=0.01, help="Validation ratio before balancing.")
    parser.add_argument("--test-ratio", type=float, default=0.20, help="Test ratio before balancing.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for custom splitting and training.")
    parser.add_argument("--max-epochs", type=int, default=None, help="Optional override for trainer.max_epochs.")
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=None,
        help="Optional override for training/validation/test dataloader batch size.",
    )
    parser.add_argument(
        "--train-num-workers",
        type=int,
        default=None,
        help="Optional override for dataloader worker count.",
    )
    parser.add_argument(
        "--serialized-dir",
        type=Path,
        default=DEFAULT_SERIALIZED_DIR,
        help="Directory used for serialized feature tensors.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help="Directory for prepared tables, checkpoints, exports, and reports.",
    )
    parser.add_argument(
        "--reuse-custom-embeddings",
        action="store_true",
        help="Reuse existing serialized embeddings for the same dataset/settings when present.",
    )
    parser.add_argument(
        "--drug-embed-batch-size",
        type=int,
        default=32,
        help="Batch size for SMILES embedding generation.",
    )
    parser.add_argument(
        "--target-embed-batch-size",
        type=int,
        default=4,
        help="Batch size for protein embedding generation.",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training and only evaluate/export from an existing checkpoint in the run directory.",
    )
    return parser.parse_args(argv)


def build_training_overrides(args: argparse.Namespace, run_root: Path, checkpoint_dir: Path) -> list[str]:
    best_param_name = args.best_param_name or default_best_param_name(args.split_strategy, args.balanced)
    overrides = [
        "tuning.param_search.tune=False",
        f"best_param_name={best_param_name}",
        f"callbacks.model_checkpoint.dirpath={normalize_dir(checkpoint_dir)}",
        "callbacks.model_checkpoint.save_top_k=1",
        "callbacks.model_checkpoint.save_last=True",
        "callbacks.model_checkpoint.save_weights_only=False",
        f"datamodule.serializer.save_path={normalize_dir(args.serialized_dir)}",
        "logger.name=custom",
        # Keep preprocess.data_path available because the current model reads it.
        f"preprocess.data_path={normalize_dir(args.artifacts_dir / 'custom_placeholder')}",
        f"hydra.run.dir={normalize_dir(run_root / 'hydra_run')}",
        "hydra.output_subdir=null",
        "hydra.job.chdir=False",
        "multiprocessing.multiprocessing=False",
        f"featurizer.drugfeaturizer.batch_size={args.drug_embed_batch_size}",
        f"featurizer.protfeaturizer.batch_size={args.target_embed_batch_size}",
        f"datamodule.splitting.splitting_strategy={normalize_split_strategy(args.split_strategy)}",
        f"datamodule.splitting.balanced={'True' if args.balanced else 'False'}",
        f"datamodule.splitting.unbalanced_ratio={args.unbalanced_ratio}",
        f"datamodule.splitting.seed={args.seed}",
        f"datamodule.splitting.ratio=[{args.train_ratio},{args.val_ratio},{args.test_ratio}]",
    ]
    if args.max_epochs is not None:
        overrides.append(f"trainer.max_epochs={args.max_epochs}")
    if args.train_batch_size is not None:
        overrides.append(f"datamodule.dm_cfg.batch_size={args.train_batch_size}")
    if args.train_num_workers is not None:
        overrides.append(f"datamodule.dm_cfg.num_workers={args.train_num_workers}")
    return overrides


def apply_explicit_training_overrides(cfg_dict: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    datamodule_cfg = cfg_dict["datamodule"]["dm_cfg"]
    if args.train_batch_size is not None:
        datamodule_cfg["batch_size"] = args.train_batch_size
    if args.train_num_workers is not None:
        datamodule_cfg["num_workers"] = args.train_num_workers
    return cfg_dict


def create_trainer(cfg_dict: dict[str, Any], tensorboard_root: Path):
    import pytorch_lightning as pl
    import torch
    from pytorch_lightning.loggers import TensorBoardLogger
    from utils import utils

    callbacks = utils.instantiate_callbacks(cfg_dict["callbacks"])
    tb_logger = TensorBoardLogger(str(tensorboard_root), name=cfg_dict["logger"]["name"])
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    return pl.Trainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=cfg_dict["trainer"]["max_epochs"],
        logger=tb_logger,
        callbacks=callbacks,
        log_every_n_steps=5,
    )


def train_custom_model(
    cfg_dict: dict[str, Any],
    dataset: dict[str, Any],
    tensorboard_root: Path,
    resume_from_checkpoint: Path | None = None,
) -> None:
    import hydra
    import pytorch_lightning as pl
    import torch

    pl.seed_everything(seed=cfg_dict["datamodule"]["splitting"]["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    data_module = hydra.utils.instantiate(cfg_dict["datamodule"], cfg_dict, dataset, _recursive_=False)
    model = hydra.utils.instantiate(cfg_dict["module"], cfg_dict, dataset, _recursive_=False)
    trainer = create_trainer(cfg_dict, tensorboard_root)
    ckpt_path = None if resume_from_checkpoint is None else str(resume_from_checkpoint)
    trainer.fit(model, data_module, ckpt_path=ckpt_path)
    trainer.validate(model, data_module)
    trainer.test(model, data_module)


def run_training_with_log(
    cfg_dict: dict[str, Any],
    dataset: dict[str, Any],
    tensorboard_root: Path,
    log_path: Path,
    argv: list[str] | None,
    command_name: str = "train-custom",
    resume_from_checkpoint: Path | None = None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(REPO_ROOT / "main.py"), command_name, *(argv or sys.argv[1:])]
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("COMMAND: " + shlex.join(command) + "\n\n")
        handle.flush()
        with contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
            train_custom_model(cfg_dict, dataset, tensorboard_root, resume_from_checkpoint=resume_from_checkpoint)


def split_assignments(dataset: dict[str, Any]):
    import pandas as pd

    assignments: dict[int, str] = {}
    for split_name in ("train", "val", "test"):
        for source_row in dataset[split_name]["source_row"].tolist():
            assignments[int(source_row)] = split_name
    return pd.Series(assignments, name="split")


def save_split_table(dataset: dict[str, Any], output_path: Path) -> Path:
    import pandas as pd

    frames = []
    for split_name in ("train", "val", "test"):
        split_frame = dataset[split_name][["source_row", "Drug_ID", "Prot_ID", "label"]].copy()
        split_frame.insert(1, "split", split_name)
        frames.append(split_frame)
    pd.concat(frames, ignore_index=True).to_csv(output_path, sep="\t", index=False)
    return output_path


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    ensure_repo_on_path()

    ratio_total = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_total - 1.0) > 1e-9:
        raise ValueError("--train-ratio + --val-ratio + --test-ratio must sum to 1.0")

    args.split_strategy = normalize_split_strategy(args.split_strategy)
    if args.best_param_name is None:
        args.best_param_name = default_best_param_name(args.split_strategy, args.balanced)

    run_name = sanitize_slug(
        f"{args.custom_data.stem}_{args.base_config}_{args.split_strategy}_{'balanced' if args.balanced else 'unbalanced'}"
    )
    run_root = args.artifacts_dir / run_name
    prepared_dir = run_root / "prepared_data"
    logs_dir = run_root / "logs"
    checkpoint_dir = run_root / "checkpoints"
    export_dir = run_root / "exports"
    tensorboard_root = run_root / "tensorboard"
    report_path = run_root / "metrics.json"
    train_log_path = logs_dir / "train.log"

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    args.serialized_dir.mkdir(parents=True, exist_ok=True)

    filtered_custom_data = read_custom_triplets(args.custom_data, args.delimiter, args.has_header)
    exclusions_report = save_exclusions_report(filtered_custom_data.excluded, prepared_dir)
    tables = build_custom_tables(filtered_custom_data.frame)
    prepared_paths = save_custom_tables(tables, prepared_dir)

    training_overrides = build_training_overrides(args, run_root, checkpoint_dir)
    cfg = compose_cfg(args.base_config, training_overrides)
    cfg_dict = update_best_params(cfg)
    cfg_dict = apply_explicit_training_overrides(cfg_dict, args)

    serializer_suffix = f"{args.base_config}_{args.split_strategy}_{'balanced' if args.balanced else 'unbalanced'}"
    custom_drug_name, custom_target_name = custom_serializer_names(args.custom_data, serializer_suffix)
    x_drug_embeddings, x_target_embeddings, embedding_paths = generate_custom_embeddings(
        cfg,
        tables,
        args.serialized_dir,
        custom_drug_name,
        custom_target_name,
        args.reuse_custom_embeddings,
    )

    relation_with_source = tables.DTI.copy()
    relation_with_source["source_row"] = filtered_custom_data.frame["source_row"].values

    from utils import utils

    dataset_for_training = utils.get_dataset(
        cfg_dict,
        x_drug_embeddings.copy(),
        x_target_embeddings.copy(),
        relation_with_source[["Drug_ID", "Prot_ID", "label", "source_row"]].copy(),
        ddi=None,
        skipped=None,
    )

    split_table_path = save_split_table(dataset_for_training, prepared_dir / "split_table.tsv")
    split_map = split_assignments(dataset_for_training)
    used_rows = int(split_map.shape[0])

    tensorboard_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TENSORBOARD_LOG_DIR", str(tensorboard_root))

    if not args.skip_training:
        run_training_with_log(cfg_dict, dataset_for_training, tensorboard_root, train_log_path, argv)

    checkpoint_path = pick_checkpoint(checkpoint_dir)

    full_prediction_dataset = build_prediction_dataset(
        x_drug_embeddings,
        x_target_embeddings,
        relation_with_source[["Drug_ID", "Prot_ID", "label", "source_row"]],
    )
    prediction_rows = predict_checkpoint_on_dataset(
        cfg_dict,
        checkpoint_path,
        full_prediction_dataset,
        relation_with_source["source_row"].astype(int).tolist(),
    )
    prediction_exports = save_prediction_export(
        filtered_custom_data,
        prediction_rows,
        export_dir,
        args.custom_data,
        args.delimiter,
        split_assignments=split_map,
    )

    test_source_rows = set(dataset_for_training["test"]["source_row"].astype(int).tolist())
    test_predictions = prediction_rows[prediction_rows["source_row"].isin(test_source_rows)].reset_index(drop=True)
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

    y_true = test_predictions["true_label"].to_numpy()
    y_score = test_predictions["probability_active"].to_numpy()
    y_pred = test_predictions["predicted_label"].to_numpy()
    metrics = {
        "test_auc": float(roc_auc_score(y_true, y_score)),
        "test_auprc": float(average_precision_score(y_true, y_score)),
        "test_f1": float(f1_score(y_true, y_pred)),
    }

    safetensors_path = export_dir / f"{checkpoint_path.stem}.safetensors"
    export_checkpoint_to_safetensors(
        checkpoint_path,
        safetensors_path,
        metadata={
            "custom_data": str(args.custom_data.resolve()),
            "base_config": args.base_config,
            "split_strategy": args.split_strategy,
            "source_checkpoint": str(checkpoint_path.resolve()),
        },
    )

    sample_count = int(len(tables.DTI))
    positive_count = int(tables.DTI["label"].sum())
    split_counts = {split_name: int(len(dataset_for_training[split_name])) for split_name in ("train", "val", "test")}
    report = {
        "mode": "custom_train_eval_export",
        "custom_data": str(args.custom_data.resolve()),
        "base_config": args.base_config,
        "best_param_name": args.best_param_name,
        "split_strategy": args.split_strategy,
        "balanced": args.balanced,
        "seed": args.seed,
        "samples": sample_count,
        "excluded_rows": int(len(filtered_custom_data.excluded)),
        "positives": positive_count,
        "negatives": sample_count - positive_count,
        "split_counts": split_counts,
        "unused_rows": sample_count - used_rows,
        "prepared_tables": {key: str(path.resolve()) for key, path in prepared_paths.items()},
        "split_table": str(split_table_path.resolve()),
        "exclusions": None if exclusions_report is None else {key: str(path.resolve()) for key, path in exclusions_report.items()},
        "custom_embeddings": {key: str(path.resolve()) for key, path in embedding_paths.items()},
        "prediction_exports": {key: str(path.resolve()) for key, path in prediction_exports.items()},
        "logs": {"train": str(train_log_path.resolve())},
        "checkpoint": str(checkpoint_path.resolve()),
        "safetensors": str(safetensors_path.resolve()),
        "metrics": metrics,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Prepared drug table: {prepared_paths['drug_table']}")
    print(f"Prepared protein table: {prepared_paths['protein_table']}")
    print(f"Prepared relation table: {prepared_paths['relation_table']}")
    print(f"Split table: {split_table_path}")
    if exclusions_report is not None:
        print(f"Excluded rows report: {exclusions_report['rows']}")
        print(f"Excluded rows summary: {exclusions_report['summary']}")
    print(f"Predictions export: {prediction_exports['csv']}")
    if "json" in prediction_exports:
        print(f"Predictions export JSON: {prediction_exports['json']}")
    print(f"Training log: {train_log_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Safetensors: {safetensors_path}")
    print(f"Custom test AUC: {metrics['test_auc']:.6f}")
    print(f"Custom test AUPRC: {metrics['test_auprc']:.6f}")
    print(f"Custom test F1: {metrics['test_f1']:.6f}")
    print(f"Metrics report: {report_path}")


if __name__ == "__main__":
    main()
