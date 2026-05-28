from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

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
from scripts.train_custom_dataset_and_export import (
    run_training_with_log,
    save_split_table,
    split_assignments,
)


DEFAULT_SERIALIZED_DIR = REPO_ROOT / "datasets" / "serialized"
DEFAULT_ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "custom_cross_dataset"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train DCGAT-DTI on one custom smiles|sequence|activation dataset, test on a different custom dataset, "
            "and export predictions plus safetensors weights."
        )
    )
    parser.add_argument("--train-data", type=Path, required=True, help="Path to the custom training dataset file.")
    parser.add_argument("--test-data", type=Path, required=True, help="Path to the custom test dataset file.")
    parser.add_argument("--train-delimiter", default="|", help="Field delimiter for CSV train input. Defaults to '|'.")
    parser.add_argument("--test-delimiter", default="|", help="Field delimiter for CSV test input. Defaults to '|'.")
    parser.add_argument("--train-has-header", action="store_true", help="Treat the first train CSV row as a header row.")
    parser.add_argument("--test-has-header", action="store_true", help="Treat the first test CSV row as a header row.")
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
        help="How to split the training dataset into train/val/test.",
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
    parser.add_argument("--test-ratio", type=float, default=0.20, help="Held-out ratio inside the training dataset before balancing.")
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
    overrides = [
        "tuning.param_search.tune=False",
        f"best_param_name={args.best_param_name}",
        f"callbacks.model_checkpoint.dirpath={normalize_dir(checkpoint_dir)}",
        "callbacks.model_checkpoint.save_top_k=1",
        "callbacks.model_checkpoint.save_last=True",
        "callbacks.model_checkpoint.save_weights_only=False",
        f"datamodule.serializer.save_path={normalize_dir(args.serialized_dir)}",
        "logger.name=custom_cross_dataset",
        f"preprocess.data_path={normalize_dir(args.artifacts_dir / 'custom_placeholder')}",
        f"hydra.run.dir={normalize_dir(run_root / 'hydra_run')}",
        "hydra.output_subdir=null",
        "hydra.job.chdir=False",
        "multiprocessing.multiprocessing=False",
        f"featurizer.drugfeaturizer.batch_size={args.drug_embed_batch_size}",
        f"featurizer.protfeaturizer.batch_size={args.target_embed_batch_size}",
        f"datamodule.splitting.splitting_strategy={args.split_strategy}",
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


def apply_explicit_training_overrides(cfg_dict: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    datamodule_cfg = cfg_dict["datamodule"]["dm_cfg"]
    if args.train_batch_size is not None:
        datamodule_cfg["batch_size"] = args.train_batch_size
    if args.train_num_workers is not None:
        datamodule_cfg["num_workers"] = args.train_num_workers
    return cfg_dict


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
        f"{args.train_data.stem}_to_{args.test_data.stem}_{args.base_config}_{args.split_strategy}_{'balanced' if args.balanced else 'unbalanced'}"
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

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    args.serialized_dir.mkdir(parents=True, exist_ok=True)

    filtered_train_data = read_custom_triplets(args.train_data, args.train_delimiter, args.train_has_header)
    filtered_test_data = read_custom_triplets(args.test_data, args.test_delimiter, args.test_has_header)

    train_exclusions_report = save_exclusions_report(filtered_train_data.excluded, train_prepared_dir)
    test_exclusions_report = save_exclusions_report(filtered_test_data.excluded, test_prepared_dir)

    train_tables = build_custom_tables(filtered_train_data.frame)
    test_tables = build_custom_tables(filtered_test_data.frame)
    train_prepared_paths = save_custom_tables(train_tables, train_prepared_dir)
    test_prepared_paths = save_custom_tables(test_tables, test_prepared_dir)

    training_overrides = build_training_overrides(args, run_root, checkpoint_dir)
    cfg = compose_cfg(args.base_config, training_overrides)
    cfg_dict = update_best_params(cfg)
    cfg_dict = apply_explicit_training_overrides(cfg_dict, args)

    train_serializer_suffix = f"train_{args.base_config}_{args.split_strategy}_{'balanced' if args.balanced else 'unbalanced'}"
    train_drug_name, train_target_name = custom_serializer_names(args.train_data, train_serializer_suffix)
    train_x_drug_embeddings, train_x_target_embeddings, train_embedding_paths = generate_custom_embeddings(
        cfg,
        train_tables,
        args.serialized_dir,
        train_drug_name,
        train_target_name,
        args.reuse_custom_embeddings,
    )

    train_relations_with_source = train_tables.DTI.copy()
    train_relations_with_source["source_row"] = filtered_train_data.frame["source_row"].values

    from utils import utils

    dataset_for_training = utils.get_dataset(
        cfg_dict,
        train_x_drug_embeddings.copy(),
        train_x_target_embeddings.copy(),
        train_relations_with_source[["Drug_ID", "Prot_ID", "label", "source_row"]].copy(),
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
            command_name="train-custom-test-custom",
        )

    checkpoint_path = pick_checkpoint(checkpoint_dir)

    test_serializer_suffix = f"test_{args.base_config}_{args.split_strategy}_{'balanced' if args.balanced else 'unbalanced'}"
    test_drug_name, test_target_name = custom_serializer_names(args.test_data, test_serializer_suffix)
    test_x_drug_embeddings, test_x_target_embeddings, test_embedding_paths = generate_custom_embeddings(
        cfg,
        test_tables,
        args.serialized_dir,
        test_drug_name,
        test_target_name,
        args.reuse_custom_embeddings,
    )

    test_relations_with_source = test_tables.DTI.copy()
    test_relations_with_source["source_row"] = filtered_test_data.frame["source_row"].values
    prediction_dataset = build_prediction_dataset(
        test_x_drug_embeddings,
        test_x_target_embeddings,
        test_relations_with_source[["Drug_ID", "Prot_ID", "label", "source_row"]],
    )
    prediction_rows = predict_checkpoint_on_dataset(
        cfg_dict,
        checkpoint_path,
        prediction_dataset,
        test_relations_with_source["source_row"].astype(int).tolist(),
    )
    prediction_exports = save_prediction_export(
        filtered_test_data,
        prediction_rows,
        export_dir,
        args.test_data,
        args.test_delimiter,
    )

    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

    y_true = prediction_rows["true_label"].to_numpy()
    y_score = prediction_rows["probability_active"].to_numpy()
    y_pred = prediction_rows["predicted_label"].to_numpy()
    metrics = {
        "external_test_auc": float(roc_auc_score(y_true, y_score)),
        "external_test_auprc": float(average_precision_score(y_true, y_score)),
        "external_test_f1": float(f1_score(y_true, y_pred)),
    }

    safetensors_path = export_dir / f"{checkpoint_path.stem}.safetensors"
    export_checkpoint_to_safetensors(
        checkpoint_path,
        safetensors_path,
        metadata={
            "train_data": str(args.train_data.resolve()),
            "test_data": str(args.test_data.resolve()),
            "base_config": args.base_config,
            "split_strategy": args.split_strategy,
            "source_checkpoint": str(checkpoint_path.resolve()),
        },
    )

    train_sample_count = int(len(train_tables.DTI))
    train_positive_count = int(train_tables.DTI["label"].sum())
    test_sample_count = int(len(test_tables.DTI))
    test_positive_count = int(test_tables.DTI["label"].sum())
    split_counts = {split_name: int(len(dataset_for_training[split_name])) for split_name in ("train", "val", "test")}
    report = {
        "mode": "custom_train_external_eval_export",
        "train_data": str(args.train_data.resolve()),
        "test_data": str(args.test_data.resolve()),
        "base_config": args.base_config,
        "best_param_name": args.best_param_name,
        "split_strategy": args.split_strategy,
        "balanced": args.balanced,
        "seed": args.seed,
        "train_samples": train_sample_count,
        "train_excluded_rows": int(len(filtered_train_data.excluded)),
        "train_positives": train_positive_count,
        "train_negatives": train_sample_count - train_positive_count,
        "test_samples": test_sample_count,
        "test_excluded_rows": int(len(filtered_test_data.excluded)),
        "test_positives": test_positive_count,
        "test_negatives": test_sample_count - test_positive_count,
        "split_counts": split_counts,
        "unused_train_rows": train_sample_count - used_rows,
        "prepared_train_tables": {key: str(path.resolve()) for key, path in train_prepared_paths.items()},
        "prepared_test_tables": {key: str(path.resolve()) for key, path in test_prepared_paths.items()},
        "split_table": str(split_table_path.resolve()),
        "train_exclusions": None if train_exclusions_report is None else {key: str(path.resolve()) for key, path in train_exclusions_report.items()},
        "test_exclusions": None if test_exclusions_report is None else {key: str(path.resolve()) for key, path in test_exclusions_report.items()},
        "train_custom_embeddings": {key: str(path.resolve()) for key, path in train_embedding_paths.items()},
        "test_custom_embeddings": {key: str(path.resolve()) for key, path in test_embedding_paths.items()},
        "prediction_exports": {key: str(path.resolve()) for key, path in prediction_exports.items()},
        "logs": {"train": str(train_log_path.resolve())},
        "checkpoint": str(checkpoint_path.resolve()),
        "safetensors": str(safetensors_path.resolve()),
        "metrics": metrics,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Prepared train drug table: {train_prepared_paths['drug_table']}")
    print(f"Prepared train protein table: {train_prepared_paths['protein_table']}")
    print(f"Prepared train relation table: {train_prepared_paths['relation_table']}")
    print(f"Prepared test drug table: {test_prepared_paths['drug_table']}")
    print(f"Prepared test protein table: {test_prepared_paths['protein_table']}")
    print(f"Prepared test relation table: {test_prepared_paths['relation_table']}")
    print(f"Train split table: {split_table_path}")
    if train_exclusions_report is not None:
        print(f"Train excluded rows report: {train_exclusions_report['rows']}")
        print(f"Train excluded rows summary: {train_exclusions_report['summary']}")
    if test_exclusions_report is not None:
        print(f"Test excluded rows report: {test_exclusions_report['rows']}")
        print(f"Test excluded rows summary: {test_exclusions_report['summary']}")
    print(f"Predictions export: {prediction_exports['csv']}")
    if "json" in prediction_exports:
        print(f"Predictions export JSON: {prediction_exports['json']}")
    print(f"Training log: {train_log_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Safetensors: {safetensors_path}")
    print(f"External test AUC: {metrics['external_test_auc']:.6f}")
    print(f"External test AUPRC: {metrics['external_test_auprc']:.6f}")
    print(f"External test F1: {metrics['external_test_f1']:.6f}")
    print(f"Metrics report: {report_path}")


if __name__ == "__main__":
    main()
