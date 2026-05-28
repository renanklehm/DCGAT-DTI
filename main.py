from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from scripts.common import (
    REPO_ROOT,
    RUN_PY,
    compose_cfg,
    default_best_param_name,
    ensure_repo_on_path,
    get_experiment,
    load_checkpoint_cfg_dict,
    normalize_split_strategy,
    pick_checkpoint,
    run_command,
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
from scripts.train_custom_dataset_and_export import (
    apply_explicit_training_overrides,
    build_training_overrides as build_custom_training_overrides,
    run_training_with_log,
    save_split_table,
    split_assignments,
)
from scripts.train_existing_scenario_and_eval_custom import (
    ensure_training_embeddings,
    scenario_training_overrides,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Common entrypoint for DCGAT-DTI utilities. Use 'reproduce-paper' for the dedicated paper workflow, "
            "or call main.py directly with the custom training/evaluation flags."
        )
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=("reproduce-paper",),
        help="Dedicated workflow command. Omit it to use the unified custom-data workflow.",
    )
    parser.add_argument("--scenario", help="Built-in scenario to train, in dataset:scenario_key form.")
    parser.add_argument("--train-data", type=Path, help="Custom training dataset path.")
    parser.add_argument(
        "--test-data",
        "--custom-data",
        dest="test_data",
        type=Path,
        help="Custom evaluation dataset path.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Existing .ckpt to evaluate on --test-data without training a new model.",
    )
    parser.add_argument("--delimiter", default="|", help="Default delimiter for CSV custom datasets.")
    parser.add_argument("--has-header", action="store_true", help="Treat custom CSV input(s) as having a header row.")
    parser.add_argument("--train-delimiter", default=None, help="Optional train CSV delimiter override.")
    parser.add_argument("--test-delimiter", default=None, help="Optional test CSV delimiter override.")
    parser.add_argument("--train-has-header", action="store_true", help="Treat the training CSV input as having a header row.")
    parser.add_argument("--test-has-header", action="store_true", help="Treat the evaluation CSV input as having a header row.")
    parser.add_argument(
        "--base-config",
        default=None,
        choices=(
            "drugbank_train_GAT.yaml",
            "bindingDB_train_GAT.yaml",
            "yamanishi_train.yaml",
            "luo_train.yaml",
        ),
        help="Hydra config baseline for custom training modes or checkpoint fallback when the checkpoint lacks cfg metadata.",
    )
    parser.add_argument(
        "--best-param-name",
        default=None,
        help="Best-params file from configs/best_params applied before custom training or checkpoint fallback config setup.",
    )
    parser.add_argument(
        "--split-strategy",
        choices=("warm", "random", "cold_drug", "cold_target", "cold_full"),
        default="warm",
        help="Split strategy for custom training data.",
    )
    parser.add_argument(
        "--balanced",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use balanced splits for custom training data.",
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
        help="Optional override for training and evaluation dataloader batch size.",
    )
    parser.add_argument(
        "--train-num-workers",
        type=int,
        default=None,
        help="Optional override for dataloader worker count in custom training modes.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Optional override for prediction batch size when exporting custom dataset scores.",
    )
    parser.add_argument(
        "--serialized-dir",
        type=Path,
        default=REPO_ROOT / "datasets" / "serialized",
        help="Directory used for serialized feature tensors.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "custom_workflow",
        help="Directory for prepared tables, checkpoints, exports, logs, and reports.",
    )
    parser.add_argument(
        "--reuse-custom-embeddings",
        action="store_true",
        help="Reuse existing serialized embeddings when present; otherwise recompute them.",
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
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        default=None,
        help="Resume custom training from a saved Lightning checkpoint.",
    )
    parser.add_argument(
        "--drugbank-root",
        type=Path,
        default=REPO_ROOT / "datasets" / "drugbank",
        help="DrugBank dataset directory used by built-in scenarios.",
    )
    parser.add_argument(
        "--bindingdb-root",
        type=Path,
        default=REPO_ROOT / "datasets" / "bindingDB",
        help="BindingDB dataset directory used by built-in scenarios.",
    )
    parser.add_argument(
        "--bindingdb-binary-threshold",
        type=float,
        help="Optional BindingDB active/inactive threshold override in nM.",
    )
    parser.add_argument(
        "--yamanishi-root",
        type=Path,
        default=REPO_ROOT / "datasets" / "yamanishi_08",
        help="Yamanishi root directory used by built-in scenarios.",
    )
    parser.add_argument(
        "--luo-root",
        type=Path,
        default=REPO_ROOT / "datasets" / "luo_dataset",
        help="Luo root directory used by built-in scenarios.",
    )
    return parser


def resolve_mode(args: argparse.Namespace) -> str:
    if args.checkpoint is not None:
        if args.scenario is not None or args.train_data is not None:
            raise ValueError("--checkpoint cannot be combined with --scenario or --train-data.")
        if args.test_data is None:
            raise ValueError("--checkpoint mode requires --test-data.")
        if args.resume_from_checkpoint is not None:
            raise ValueError("--resume-from-checkpoint is only valid for custom training modes.")
        return "checkpoint_eval"

    if args.scenario is not None:
        if args.train_data is not None:
            raise ValueError("--scenario cannot be combined with --train-data.")
        if args.test_data is None:
            raise ValueError("--scenario mode requires --test-data.")
        if args.resume_from_checkpoint is not None:
            raise ValueError("--resume-from-checkpoint is only valid for custom training modes.")
        return "scenario_eval"

    if args.train_data is not None:
        return "custom_cross_eval" if args.test_data is not None else "custom_split_eval"

    raise ValueError("Provide one of --scenario, --train-data, or --checkpoint.")


def validate_ratios(args: argparse.Namespace) -> None:
    ratio_total = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_total - 1.0) > 1e-9:
        raise ValueError("--train-ratio + --val-ratio + --test-ratio must sum to 1.0")


def resolve_base_config(args: argparse.Namespace) -> str:
    return args.base_config or "drugbank_train_GAT.yaml"


def resolve_delimiter(args: argparse.Namespace, kind: str) -> str:
    specific = getattr(args, f"{kind}_delimiter")
    return args.delimiter if specific is None else specific


def resolve_has_header(args: argparse.Namespace, kind: str) -> bool:
    return args.has_header or bool(getattr(args, f"{kind}_has_header"))


def prepare_custom_dataset(path: Path, delimiter: str, has_header: bool, prepared_dir: Path) -> dict[str, Any]:
    filtered_data = read_custom_triplets(path, delimiter, has_header)
    exclusions_report = save_exclusions_report(filtered_data.excluded, prepared_dir)
    tables = build_custom_tables(filtered_data.frame)
    prepared_paths = save_custom_tables(tables, prepared_dir)
    relations_with_source = tables.DTI.copy()
    relations_with_source["source_row"] = filtered_data.frame["source_row"].values
    return {
        "filtered": filtered_data,
        "tables": tables,
        "prepared_paths": prepared_paths,
        "exclusions_report": exclusions_report,
        "relations_with_source": relations_with_source,
    }


def apply_runtime_overrides(cfg_dict: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.eval_batch_size is not None:
        cfg_dict["datamodule"]["dm_cfg"]["batch_size"] = args.eval_batch_size
    elif args.train_batch_size is not None:
        cfg_dict["datamodule"]["dm_cfg"]["batch_size"] = args.train_batch_size

    cfg_dict["featurizer"]["drugfeaturizer"]["batch_size"] = args.drug_embed_batch_size
    cfg_dict["featurizer"]["protfeaturizer"]["batch_size"] = args.target_embed_batch_size
    return cfg_dict


def checkpoint_fallback_overrides(args: argparse.Namespace) -> list[str]:
    overrides = [
        "tuning.param_search.tune=False",
        f"best_param_name={args.best_param_name or default_best_param_name(args.split_strategy, args.balanced)}",
        f"datamodule.serializer.save_path={args.serialized_dir.resolve().as_posix().rstrip('/')}/",
        "multiprocessing.multiprocessing=False",
        f"featurizer.drugfeaturizer.batch_size={args.drug_embed_batch_size}",
        f"featurizer.protfeaturizer.batch_size={args.target_embed_batch_size}",
    ]
    if args.eval_batch_size is not None:
        overrides.append(f"datamodule.dm_cfg.batch_size={args.eval_batch_size}")
    elif args.train_batch_size is not None:
        overrides.append(f"datamodule.dm_cfg.batch_size={args.train_batch_size}")
    return overrides


def load_or_compose_checkpoint_cfg(args: argparse.Namespace, checkpoint_path: Path) -> tuple[Any, dict[str, Any]]:
    from omegaconf import OmegaConf

    cfg_dict = load_checkpoint_cfg_dict(checkpoint_path)
    if cfg_dict is None:
        if args.base_config is None:
            raise ValueError(
                "The checkpoint does not contain a usable training config. Pass --base-config to supply a fallback."
            )
        cfg = compose_cfg(args.base_config, checkpoint_fallback_overrides(args))
        cfg_dict = update_best_params(cfg)

    cfg_dict = apply_runtime_overrides(cfg_dict, args)
    cfg = OmegaConf.create(cfg_dict)
    return cfg, cfg_dict


def evaluate_relations(
    cfg: Any,
    cfg_dict: dict[str, Any],
    checkpoint_path: Path,
    serialized_dir: Path,
    dataset_path: Path,
    serializer_suffix: str,
    tables: Any,
    relations_with_source: Any,
    reuse_custom_embeddings: bool,
) -> tuple[Any, dict[str, Path], Any]:
    drug_name, target_name = custom_serializer_names(dataset_path, serializer_suffix)
    x_drug_embeddings, x_target_embeddings, embedding_paths = generate_custom_embeddings(
        cfg,
        tables,
        serialized_dir,
        drug_name,
        target_name,
        reuse_custom_embeddings,
    )
    prediction_dataset = build_prediction_dataset(
        x_drug_embeddings,
        x_target_embeddings,
        relations_with_source[["Drug_ID", "Prot_ID", "label", "source_row"]],
    )
    prediction_rows = predict_checkpoint_on_dataset(
        cfg_dict,
        checkpoint_path,
        prediction_dataset,
        relations_with_source["source_row"].astype(int).tolist(),
    )
    return prediction_rows, embedding_paths, prediction_dataset


def compute_metrics(prediction_rows: Any) -> tuple[dict[str, float | None], list[str]]:
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

    y_true = prediction_rows["true_label"].to_numpy()
    y_score = prediction_rows["probability_active"].to_numpy()
    y_pred = prediction_rows["predicted_label"].to_numpy()

    metrics: dict[str, float | None] = {}
    notes: list[str] = []
    try:
        metrics["auc"] = float(roc_auc_score(y_true, y_score))
    except ValueError:
        metrics["auc"] = None
        notes.append("AUC is undefined because the evaluation set contains a single class.")

    try:
        metrics["auprc"] = float(average_precision_score(y_true, y_score))
    except ValueError:
        metrics["auprc"] = None
        notes.append("AUPRC is undefined for the supplied evaluation labels.")

    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    return metrics, notes


def write_report(report_path: Path, report: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}"


def print_common_summary(
    prepared_paths: dict[str, Path],
    exclusions_report: dict[str, Path] | None,
    prediction_exports: dict[str, Path],
    checkpoint_path: Path,
    report_path: Path,
    metrics: dict[str, float | None],
    safetensors_path: Path | None = None,
    split_table_path: Path | None = None,
    prefix: str = "",
) -> None:
    label_prefix = f"{prefix} " if prefix else ""
    print(f"Prepared {label_prefix}drug table: {prepared_paths['drug_table']}")
    print(f"Prepared {label_prefix}protein table: {prepared_paths['protein_table']}")
    print(f"Prepared {label_prefix}relation table: {prepared_paths['relation_table']}")
    if split_table_path is not None:
        print(f"{label_prefix.capitalize()}split table: {split_table_path}")
    if exclusions_report is not None:
        print(f"{label_prefix.capitalize()}excluded rows report: {exclusions_report['rows']}")
        print(f"{label_prefix.capitalize()}excluded rows summary: {exclusions_report['summary']}")
    print(f"Predictions export: {prediction_exports['csv']}")
    if "json" in prediction_exports:
        print(f"Predictions export JSON: {prediction_exports['json']}")
    print(f"Checkpoint: {checkpoint_path}")
    if safetensors_path is not None:
        print(f"Safetensors: {safetensors_path}")
    print(f"Evaluation AUC: {format_metric(metrics['auc'])}")
    print(f"Evaluation AUPRC: {format_metric(metrics['auprc'])}")
    print(f"Evaluation F1: {format_metric(metrics['f1'])}")
    print(f"Metrics report: {report_path}")


def run_custom_split_eval(args: argparse.Namespace, argv: list[str] | None) -> None:
    from utils import utils

    args.split_strategy = normalize_split_strategy(args.split_strategy)
    if args.best_param_name is None:
        args.best_param_name = default_best_param_name(args.split_strategy, args.balanced)

    base_config = resolve_base_config(args)
    run_name = sanitize_slug(
        f"{args.train_data.stem}_{base_config}_{args.split_strategy}_{'balanced' if args.balanced else 'unbalanced'}"
    )
    run_root = args.artifacts_dir / run_name
    prepared_dir = run_root / "prepared_data"
    logs_dir = run_root / "logs"
    checkpoint_dir = run_root / "checkpoints"
    export_dir = run_root / "exports"
    tensorboard_root = run_root / "tensorboard"
    report_path = run_root / "metrics.json"
    train_log_path = logs_dir / "train.log"

    prepared = prepare_custom_dataset(
        args.train_data,
        resolve_delimiter(args, "train"),
        resolve_has_header(args, "train"),
        prepared_dir,
    )

    training_overrides = build_custom_training_overrides(args, run_root, checkpoint_dir)
    cfg = compose_cfg(base_config, training_overrides)
    cfg_dict = update_best_params(cfg)
    cfg_dict = apply_explicit_training_overrides(cfg_dict, args)
    cfg_dict = apply_runtime_overrides(cfg_dict, args)

    serializer_suffix = f"{base_config}_{args.split_strategy}_{'balanced' if args.balanced else 'unbalanced'}"
    drug_name, target_name = custom_serializer_names(args.train_data, serializer_suffix)
    x_drug_embeddings, x_target_embeddings, embedding_paths = generate_custom_embeddings(
        cfg,
        prepared["tables"],
        args.serialized_dir,
        drug_name,
        target_name,
        args.reuse_custom_embeddings,
    )

    dataset_for_training = utils.get_dataset(
        cfg_dict,
        x_drug_embeddings.copy(),
        x_target_embeddings.copy(),
        prepared["relations_with_source"][["Drug_ID", "Prot_ID", "label", "source_row"]].copy(),
        ddi=None,
        skipped=None,
    )

    split_table_path = save_split_table(dataset_for_training, prepared_dir / "split_table.tsv")
    split_map = split_assignments(dataset_for_training)
    used_rows = int(split_map.shape[0])

    tensorboard_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TENSORBOARD_LOG_DIR", str(tensorboard_root))

    if not args.skip_training:
        run_training_with_log(
            cfg_dict,
            dataset_for_training,
            tensorboard_root,
            train_log_path,
            argv,
            resume_from_checkpoint=args.resume_from_checkpoint,
        )

    checkpoint_path = pick_checkpoint(checkpoint_dir)
    full_prediction_dataset = build_prediction_dataset(
        x_drug_embeddings,
        x_target_embeddings,
        prepared["relations_with_source"][["Drug_ID", "Prot_ID", "label", "source_row"]],
    )
    prediction_rows = predict_checkpoint_on_dataset(
        cfg_dict,
        checkpoint_path,
        full_prediction_dataset,
        prepared["relations_with_source"]["source_row"].astype(int).tolist(),
    )
    prediction_exports = save_prediction_export(
        prepared["filtered"],
        prediction_rows,
        export_dir,
        args.train_data,
        resolve_delimiter(args, "train"),
        split_assignments=split_map,
    )

    test_source_rows = set(dataset_for_training["test"]["source_row"].astype(int).tolist())
    test_predictions = prediction_rows[prediction_rows["source_row"].isin(test_source_rows)].reset_index(drop=True)
    metrics, metric_notes = compute_metrics(test_predictions)

    safetensors_path = export_dir / f"{checkpoint_path.stem}.safetensors"
    export_checkpoint_to_safetensors(
        checkpoint_path,
        safetensors_path,
        metadata={
            "custom_data": str(args.train_data.resolve()),
            "base_config": base_config,
            "split_strategy": args.split_strategy,
            "source_checkpoint": str(checkpoint_path.resolve()),
        },
    )

    sample_count = int(len(prepared["tables"].DTI))
    positive_count = int(prepared["tables"].DTI["label"].sum())
    split_counts = {split_name: int(len(dataset_for_training[split_name])) for split_name in ("train", "val", "test")}
    report = {
        "mode": "custom_split_eval",
        "train_data": str(args.train_data.resolve()),
        "base_config": base_config,
        "best_param_name": args.best_param_name,
        "split_strategy": args.split_strategy,
        "balanced": args.balanced,
        "seed": args.seed,
        "samples": sample_count,
        "excluded_rows": int(len(prepared["filtered"].excluded)),
        "positives": positive_count,
        "negatives": sample_count - positive_count,
        "split_counts": split_counts,
        "unused_rows": sample_count - used_rows,
        "prepared_tables": {key: str(path.resolve()) for key, path in prepared["prepared_paths"].items()},
        "split_table": str(split_table_path.resolve()),
        "exclusions": None
        if prepared["exclusions_report"] is None
        else {key: str(path.resolve()) for key, path in prepared["exclusions_report"].items()},
        "custom_embeddings": {key: str(path.resolve()) for key, path in embedding_paths.items()},
        "prediction_exports": {key: str(path.resolve()) for key, path in prediction_exports.items()},
        "logs": {"train": str(train_log_path.resolve())},
        "checkpoint": str(checkpoint_path.resolve()),
        "safetensors": str(safetensors_path.resolve()),
        "metrics": metrics,
        "metric_notes": metric_notes,
    }
    write_report(report_path, report)

    print(f"Training log: {train_log_path}")
    print_common_summary(
        prepared["prepared_paths"],
        prepared["exclusions_report"],
        prediction_exports,
        checkpoint_path,
        report_path,
        metrics,
        safetensors_path=safetensors_path,
        split_table_path=split_table_path,
    )


def run_custom_cross_eval(args: argparse.Namespace, argv: list[str] | None) -> None:
    from utils import utils

    args.split_strategy = normalize_split_strategy(args.split_strategy)
    if args.best_param_name is None:
        args.best_param_name = default_best_param_name(args.split_strategy, args.balanced)

    base_config = resolve_base_config(args)
    run_name = sanitize_slug(
        f"{args.train_data.stem}_to_{args.test_data.stem}_{base_config}_{args.split_strategy}_{'balanced' if args.balanced else 'unbalanced'}"
    )
    run_root = args.artifacts_dir / run_name
    train_prepared_dir = run_root / "prepared_train_data"
    test_prepared_dir = run_root / "prepared_test_data"
    logs_dir = run_root / "logs"
    checkpoint_dir = run_root / "checkpoints"
    export_dir = run_root / "exports"
    tensorboard_root = run_root / "tensorboard"
    report_path = run_root / "metrics.json"
    train_log_path = logs_dir / "train.log"

    train_prepared = prepare_custom_dataset(
        args.train_data,
        resolve_delimiter(args, "train"),
        resolve_has_header(args, "train"),
        train_prepared_dir,
    )
    test_prepared = prepare_custom_dataset(
        args.test_data,
        resolve_delimiter(args, "test"),
        resolve_has_header(args, "test"),
        test_prepared_dir,
    )

    training_overrides = build_custom_training_overrides(args, run_root, checkpoint_dir)
    cfg = compose_cfg(base_config, training_overrides)
    cfg_dict = update_best_params(cfg)
    cfg_dict = apply_explicit_training_overrides(cfg_dict, args)
    cfg_dict = apply_runtime_overrides(cfg_dict, args)

    train_serializer_suffix = f"train_{base_config}_{args.split_strategy}_{'balanced' if args.balanced else 'unbalanced'}"
    train_drug_name, train_target_name = custom_serializer_names(args.train_data, train_serializer_suffix)
    train_x_drug_embeddings, train_x_target_embeddings, train_embedding_paths = generate_custom_embeddings(
        cfg,
        train_prepared["tables"],
        args.serialized_dir,
        train_drug_name,
        train_target_name,
        args.reuse_custom_embeddings,
    )

    dataset_for_training = utils.get_dataset(
        cfg_dict,
        train_x_drug_embeddings.copy(),
        train_x_target_embeddings.copy(),
        train_prepared["relations_with_source"][["Drug_ID", "Prot_ID", "label", "source_row"]].copy(),
        ddi=None,
        skipped=None,
    )

    split_table_path = save_split_table(dataset_for_training, train_prepared_dir / "split_table.tsv")
    split_map = split_assignments(dataset_for_training)
    used_rows = int(split_map.shape[0])

    tensorboard_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TENSORBOARD_LOG_DIR", str(tensorboard_root))

    if not args.skip_training:
        run_training_with_log(
            cfg_dict,
            dataset_for_training,
            tensorboard_root,
            train_log_path,
            argv,
            resume_from_checkpoint=args.resume_from_checkpoint,
        )

    checkpoint_path = pick_checkpoint(checkpoint_dir)
    test_serializer_suffix = f"test_{base_config}_{args.split_strategy}_{'balanced' if args.balanced else 'unbalanced'}"
    prediction_rows, test_embedding_paths, _ = evaluate_relations(
        cfg,
        cfg_dict,
        checkpoint_path,
        args.serialized_dir,
        args.test_data,
        test_serializer_suffix,
        test_prepared["tables"],
        test_prepared["relations_with_source"],
        args.reuse_custom_embeddings,
    )
    prediction_exports = save_prediction_export(
        test_prepared["filtered"],
        prediction_rows,
        export_dir,
        args.test_data,
        resolve_delimiter(args, "test"),
    )
    metrics, metric_notes = compute_metrics(prediction_rows)

    safetensors_path = export_dir / f"{checkpoint_path.stem}.safetensors"
    export_checkpoint_to_safetensors(
        checkpoint_path,
        safetensors_path,
        metadata={
            "train_data": str(args.train_data.resolve()),
            "test_data": str(args.test_data.resolve()),
            "base_config": base_config,
            "split_strategy": args.split_strategy,
            "source_checkpoint": str(checkpoint_path.resolve()),
        },
    )

    train_sample_count = int(len(train_prepared["tables"].DTI))
    train_positive_count = int(train_prepared["tables"].DTI["label"].sum())
    test_sample_count = int(len(test_prepared["tables"].DTI))
    test_positive_count = int(test_prepared["tables"].DTI["label"].sum())
    split_counts = {split_name: int(len(dataset_for_training[split_name])) for split_name in ("train", "val", "test")}
    report = {
        "mode": "custom_cross_eval",
        "train_data": str(args.train_data.resolve()),
        "test_data": str(args.test_data.resolve()),
        "base_config": base_config,
        "best_param_name": args.best_param_name,
        "split_strategy": args.split_strategy,
        "balanced": args.balanced,
        "seed": args.seed,
        "train_samples": train_sample_count,
        "train_excluded_rows": int(len(train_prepared["filtered"].excluded)),
        "train_positives": train_positive_count,
        "train_negatives": train_sample_count - train_positive_count,
        "test_samples": test_sample_count,
        "test_excluded_rows": int(len(test_prepared["filtered"].excluded)),
        "test_positives": test_positive_count,
        "test_negatives": test_sample_count - test_positive_count,
        "split_counts": split_counts,
        "unused_train_rows": train_sample_count - used_rows,
        "prepared_train_tables": {key: str(path.resolve()) for key, path in train_prepared["prepared_paths"].items()},
        "prepared_test_tables": {key: str(path.resolve()) for key, path in test_prepared["prepared_paths"].items()},
        "split_table": str(split_table_path.resolve()),
        "train_exclusions": None
        if train_prepared["exclusions_report"] is None
        else {key: str(path.resolve()) for key, path in train_prepared["exclusions_report"].items()},
        "test_exclusions": None
        if test_prepared["exclusions_report"] is None
        else {key: str(path.resolve()) for key, path in test_prepared["exclusions_report"].items()},
        "train_custom_embeddings": {key: str(path.resolve()) for key, path in train_embedding_paths.items()},
        "test_custom_embeddings": {key: str(path.resolve()) for key, path in test_embedding_paths.items()},
        "prediction_exports": {key: str(path.resolve()) for key, path in prediction_exports.items()},
        "logs": {"train": str(train_log_path.resolve())},
        "checkpoint": str(checkpoint_path.resolve()),
        "safetensors": str(safetensors_path.resolve()),
        "metrics": metrics,
        "metric_notes": metric_notes,
    }
    write_report(report_path, report)

    print(f"Training log: {train_log_path}")
    print_common_summary(
        test_prepared["prepared_paths"],
        test_prepared["exclusions_report"],
        prediction_exports,
        checkpoint_path,
        report_path,
        metrics,
        safetensors_path=safetensors_path,
        prefix="test",
    )


def run_scenario_eval(args: argparse.Namespace) -> None:
    from omegaconf import OmegaConf

    experiment = get_experiment(args.scenario)
    scenario_slug = sanitize_slug(f"{experiment.dataset}_{experiment.scenario_key}")
    custom_slug = sanitize_slug(args.test_data.stem)
    run_root = args.artifacts_dir / f"{custom_slug}__{scenario_slug}"
    prepared_dir = run_root / "prepared_data"
    logs_dir = run_root / "logs"
    checkpoint_dir = run_root / "checkpoints"
    export_dir = run_root / "exports"
    report_path = run_root / "metrics.json"
    train_log_path = logs_dir / "train.log"

    prepared = prepare_custom_dataset(
        args.test_data,
        resolve_delimiter(args, "test"),
        resolve_has_header(args, "test"),
        prepared_dir,
    )

    run_tag = "single_run"
    training_overrides = scenario_training_overrides(
        args,
        experiment.dataset,
        experiment.config_name,
        experiment.best_param_name,
        experiment.overrides,
        run_tag,
        checkpoint_dir,
    )
    ensure_training_embeddings(experiment.config_name, training_overrides, run_root)

    if not args.skip_training:
        command = [sys.executable, str(RUN_PY), "--config-name", experiment.config_name, *training_overrides]
        run_command(command, train_log_path)

    checkpoint_path = pick_checkpoint(checkpoint_dir)
    cfg = compose_cfg(experiment.config_name, training_overrides)
    cfg_dict = update_best_params(cfg)
    cfg_dict = apply_runtime_overrides(cfg_dict, args)
    cfg = OmegaConf.create(cfg_dict)

    prediction_rows, embedding_paths, _ = evaluate_relations(
        cfg,
        cfg_dict,
        checkpoint_path,
        args.serialized_dir,
        args.test_data,
        args.scenario.replace(":", "_"),
        prepared["tables"],
        prepared["relations_with_source"],
        args.reuse_custom_embeddings,
    )
    prediction_exports = save_prediction_export(
        prepared["filtered"],
        prediction_rows,
        export_dir,
        args.test_data,
        resolve_delimiter(args, "test"),
    )
    metrics, metric_notes = compute_metrics(prediction_rows)

    safetensors_path = export_dir / f"{checkpoint_path.stem}.safetensors"
    export_checkpoint_to_safetensors(
        checkpoint_path,
        safetensors_path,
        metadata={
            "scenario": args.scenario,
            "source_checkpoint": str(checkpoint_path.resolve()),
            "custom_data": str(args.test_data.resolve()),
        },
    )

    positive_count = int(prepared["tables"].DTI["label"].sum())
    sample_count = int(len(prepared["tables"].DTI))
    report = {
        "mode": "scenario_eval",
        "scenario": args.scenario,
        "trained_on_builtin_dataset": experiment.dataset,
        "best_param_name": experiment.best_param_name,
        "bindingdb_binary_threshold": args.bindingdb_binary_threshold,
        "test_data": str(args.test_data.resolve()),
        "samples": sample_count,
        "excluded_rows": int(len(prepared["filtered"].excluded)),
        "positives": positive_count,
        "negatives": sample_count - positive_count,
        "prepared_tables": {key: str(path.resolve()) for key, path in prepared["prepared_paths"].items()},
        "exclusions": None
        if prepared["exclusions_report"] is None
        else {key: str(path.resolve()) for key, path in prepared["exclusions_report"].items()},
        "custom_embeddings": {key: str(path.resolve()) for key, path in embedding_paths.items()},
        "prediction_exports": {key: str(path.resolve()) for key, path in prediction_exports.items()},
        "checkpoint": str(checkpoint_path.resolve()),
        "safetensors": str(safetensors_path.resolve()),
        "metrics": metrics,
        "metric_notes": metric_notes,
    }
    if train_log_path.exists():
        report["logs"] = {"train": str(train_log_path.resolve())}
        print(f"Training log: {train_log_path}")
    write_report(report_path, report)

    print_common_summary(
        prepared["prepared_paths"],
        prepared["exclusions_report"],
        prediction_exports,
        checkpoint_path,
        report_path,
        metrics,
        safetensors_path=safetensors_path,
    )


def run_checkpoint_eval(args: argparse.Namespace) -> None:
    checkpoint_path = args.checkpoint.resolve()
    run_name = sanitize_slug(f"{args.test_data.stem}__{checkpoint_path.stem}")
    run_root = args.artifacts_dir / run_name
    prepared_dir = run_root / "prepared_data"
    export_dir = run_root / "exports"
    report_path = run_root / "metrics.json"

    prepared = prepare_custom_dataset(
        args.test_data,
        resolve_delimiter(args, "test"),
        resolve_has_header(args, "test"),
        prepared_dir,
    )

    cfg, cfg_dict = load_or_compose_checkpoint_cfg(args, checkpoint_path)
    prediction_rows, embedding_paths, _ = evaluate_relations(
        cfg,
        cfg_dict,
        checkpoint_path,
        args.serialized_dir,
        args.test_data,
        f"checkpoint_{checkpoint_path.stem}",
        prepared["tables"],
        prepared["relations_with_source"],
        args.reuse_custom_embeddings,
    )
    prediction_exports = save_prediction_export(
        prepared["filtered"],
        prediction_rows,
        export_dir,
        args.test_data,
        resolve_delimiter(args, "test"),
    )
    metrics, metric_notes = compute_metrics(prediction_rows)

    report = {
        "mode": "checkpoint_eval",
        "checkpoint": str(checkpoint_path),
        "test_data": str(args.test_data.resolve()),
        "samples": int(len(prepared["tables"].DTI)),
        "excluded_rows": int(len(prepared["filtered"].excluded)),
        "prepared_tables": {key: str(path.resolve()) for key, path in prepared["prepared_paths"].items()},
        "exclusions": None
        if prepared["exclusions_report"] is None
        else {key: str(path.resolve()) for key, path in prepared["exclusions_report"].items()},
        "custom_embeddings": {key: str(path.resolve()) for key, path in embedding_paths.items()},
        "prediction_exports": {key: str(path.resolve()) for key, path in prediction_exports.items()},
        "metrics": metrics,
        "metric_notes": metric_notes,
    }
    write_report(report_path, report)

    print_common_summary(
        prepared["prepared_paths"],
        prepared["exclusions_report"],
        prediction_exports,
        checkpoint_path,
        report_path,
        metrics,
    )


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    if not argv or argv[0] in {"-h", "--help"}:
        parser.print_help()
        return

    if argv[0] == "reproduce-paper":
        importlib.import_module("scripts.reproduce_paper").main(argv[1:])
        return

    args = parser.parse_args(argv)
    ensure_repo_on_path()
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    args.serialized_dir.mkdir(parents=True, exist_ok=True)

    mode = resolve_mode(args)
    if mode in {"custom_split_eval", "custom_cross_eval"}:
        validate_ratios(args)

    if mode == "custom_split_eval":
        run_custom_split_eval(args, argv)
        return
    if mode == "custom_cross_eval":
        run_custom_cross_eval(args, argv)
        return
    if mode == "scenario_eval":
        run_scenario_eval(args)
        return
    if mode == "checkpoint_eval":
        run_checkpoint_eval(args)
        return

    raise RuntimeError(f"Unsupported workflow mode: {mode}")


if __name__ == "__main__":
    main()
