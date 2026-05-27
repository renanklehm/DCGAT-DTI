from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scripts.common import (
    REPO_ROOT,
    RUN_PY,
    compose_cfg,
    ensure_repo_on_path,
    get_experiment,
    normalize_dir,
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


DEFAULT_SERIALIZED_DIR = REPO_ROOT / "datasets" / "serialized"
DEFAULT_ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "custom_eval"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train one existing paper scenario on a built-in dataset, export the trained weights to safetensors, "
            "and evaluate them on a custom smiles|sequence|activation file."
        )
    )
    parser.add_argument("--custom-data", type=Path, required=True, help="Path to the custom dataset file.")
    parser.add_argument(
        "--scenario",
        required=True,
        help="Scenario to train, in dataset:scenario_key form. Example: drugbank:balanced_warm.",
    )
    parser.add_argument("--delimiter", default="|", help="Field delimiter for the custom dataset file.")
    parser.add_argument("--has-header", action="store_true", help="Treat the first row as a header row.")
    parser.add_argument("--seed", type=int, default=42, help="Split seed for DrugBank/BindingDB scenarios.")
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
        help="Directory for prepared tables, logs, checkpoints, safetensors, and reports.",
    )
    parser.add_argument(
        "--drugbank-root",
        type=Path,
        default=REPO_ROOT / "datasets" / "drugbank",
        help="DrugBank dataset directory used by the built-in scenario.",
    )
    parser.add_argument(
        "--bindingdb-root",
        type=Path,
        default=REPO_ROOT / "datasets" / "bindingDB",
        help="BindingDB dataset directory used by the built-in scenario.",
    )
    parser.add_argument(
        "--bindingdb-binary-threshold",
        type=float,
        help=(
            "Override BindingDB binary labeling with a single cutoff in nM. "
            "Pairs with affinity <= cutoff become active and > cutoff become inactive."
        ),
    )
    parser.add_argument(
        "--yamanishi-root",
        type=Path,
        default=REPO_ROOT / "datasets" / "yamanishi_08",
        help="Yamanishi root directory used by the built-in scenario.",
    )
    parser.add_argument(
        "--luo-root",
        type=Path,
        default=REPO_ROOT / "datasets" / "luo_dataset",
        help="Luo root directory used by the built-in scenario.",
    )
    parser.add_argument(
        "--reuse-custom-embeddings",
        action="store_true",
        help="Reuse existing custom serialized embeddings when present.",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training and only run export/evaluation. Requires an existing checkpoint.",
    )
    return parser.parse_args(argv)


def scenario_training_overrides(
    args: argparse.Namespace,
    dataset: str,
    config_name: str,
    best_param_name: str,
    experiment_overrides: tuple[str, ...],
    run_tag: str,
    checkpoint_dir: Path,
) -> list[str]:
    overrides = [
        "tuning.param_search.tune=False",
        f"best_param_name={best_param_name}",
        f"callbacks.model_checkpoint.dirpath={normalize_dir(checkpoint_dir)}",
        "callbacks.model_checkpoint.save_top_k=1",
        "callbacks.model_checkpoint.save_last=True",
        "callbacks.model_checkpoint.save_weights_only=False",
        f"datamodule.serializer.save_path={normalize_dir(args.serialized_dir)}",
        f"logger.name={dataset}",
        f"hydra.run.dir={normalize_dir(args.artifacts_dir / 'hydra_runs' / dataset / run_tag)}",
        "hydra.output_subdir=null",
        "hydra.job.chdir=False",
    ]

    if dataset == "drugbank":
        overrides.append(f"preprocess.data_path={normalize_dir(args.drugbank_root)}")
    elif dataset == "bindingDB":
        overrides.append(f"preprocess.data_path={normalize_dir(args.bindingdb_root)}")
        if args.bindingdb_binary_threshold is not None:
            overrides.append(f"preprocess.threshold={args.bindingdb_binary_threshold}")
    elif dataset == "yamanishi":
        overrides.append(f"preprocess.root_path={normalize_dir(args.yamanishi_root)}")
        overrides.append("multiprocessing.multiprocessing=True")
        overrides.append("multiprocessing.num_process=10")
        overrides.append("multiprocessing.concurrent_process=1")
    elif dataset == "luo":
        overrides.append(f"preprocess.root_path={normalize_dir(args.luo_root)}")
        overrides.append("multiprocessing.multiprocessing=True")
        overrides.append("multiprocessing.num_process=10")
        overrides.append("multiprocessing.concurrent_process=1")

    overrides.extend(experiment_overrides)

    if dataset in {"drugbank", "bindingDB"}:
        overrides.append(f"datamodule.splitting.seed={args.seed}")

    return overrides


def ensure_training_embeddings(config_name: str, training_overrides: list[str], artifacts_root: Path) -> None:
    from scripts.common import resolve_training_serializer_paths

    cfg = compose_cfg(config_name, training_overrides)
    drug_path, target_path = resolve_training_serializer_paths(cfg)
    if drug_path.exists() and target_path.exists():
        return

    bootstrap_overrides = [
        *training_overrides,
        "datamodule.serializer.load_serialized=False",
        "multiprocessing.multiprocessing=False",
    ]
    bootstrap_log = artifacts_root / "logs" / "feature_bootstrap.log"
    command = [sys.executable, str(RUN_PY), "--config-name", config_name, *bootstrap_overrides]
    run_command(command, bootstrap_log)

    if not drug_path.exists() or not target_path.exists():
        raise FileNotFoundError(f"Built-in scenario embeddings were not created as expected: {drug_path}, {target_path}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    ensure_repo_on_path()

    experiment = get_experiment(args.scenario)
    scenario_slug = sanitize_slug(f"{experiment.dataset}_{experiment.scenario_key}")
    custom_slug = sanitize_slug(args.custom_data.stem)
    run_root = args.artifacts_dir / f"{custom_slug}__{scenario_slug}"
    prepared_dir = run_root / "prepared_data"
    logs_dir = run_root / "logs"
    checkpoint_dir = run_root / "checkpoints"
    export_dir = run_root / "exports"
    report_path = run_root / "metrics.json"

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    args.serialized_dir.mkdir(parents=True, exist_ok=True)

    filtered_custom_data = read_custom_triplets(args.custom_data, args.delimiter, args.has_header)
    exclusions_report = save_exclusions_report(filtered_custom_data.excluded, prepared_dir)
    tables = build_custom_tables(filtered_custom_data.frame)
    prepared_paths = save_custom_tables(tables, prepared_dir)

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
        train_log_path = logs_dir / "train.log"
        command = [sys.executable, str(RUN_PY), "--config-name", experiment.config_name, *training_overrides]
        run_command(command, train_log_path)

    checkpoint_path = pick_checkpoint(checkpoint_dir)
    eval_cfg = compose_cfg(experiment.config_name, training_overrides)
    eval_cfg_dict = update_best_params(eval_cfg)

    custom_drug_name, custom_target_name = custom_serializer_names(args.custom_data, args.scenario.replace(":", "_"))
    x_drug_embeddings, x_target_embeddings, embedding_paths = generate_custom_embeddings(
        eval_cfg,
        tables,
        args.serialized_dir,
        custom_drug_name,
        custom_target_name,
        args.reuse_custom_embeddings,
    )

    relations_with_source = tables.DTI.copy()
    relations_with_source["source_row"] = filtered_custom_data.frame["source_row"].values
    prediction_dataset = build_prediction_dataset(
        x_drug_embeddings,
        x_target_embeddings,
        relations_with_source[["Drug_ID", "Prot_ID", "label", "source_row"]],
    )
    prediction_rows = predict_checkpoint_on_dataset(
        eval_cfg_dict,
        checkpoint_path,
        prediction_dataset,
        relations_with_source["source_row"].astype(int).tolist(),
    )
    prediction_exports = save_prediction_export(
        filtered_custom_data,
        prediction_rows,
        export_dir,
        args.custom_data,
        args.delimiter,
    )

    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

    y_true = prediction_rows["true_label"].to_numpy()
    y_score = prediction_rows["probability_active"].to_numpy()
    y_pred = prediction_rows["predicted_label"].to_numpy()
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
            "scenario": args.scenario,
            "source_checkpoint": str(checkpoint_path.resolve()),
            "custom_data": str(args.custom_data.resolve()),
        },
    )

    positive_count = int(tables.DTI["label"].sum())
    sample_count = int(len(tables.DTI))
    report = {
        "scenario": args.scenario,
        "trained_on_builtin_dataset": experiment.dataset,
        "best_param_name": experiment.best_param_name,
        "bindingdb_binary_threshold": args.bindingdb_binary_threshold,
        "custom_data": str(args.custom_data.resolve()),
        "samples": sample_count,
        "excluded_rows": int(len(filtered_custom_data.excluded)),
        "positives": positive_count,
        "negatives": sample_count - positive_count,
        "prepared_tables": {key: str(path.resolve()) for key, path in prepared_paths.items()},
        "exclusions": None if exclusions_report is None else {key: str(path.resolve()) for key, path in exclusions_report.items()},
        "custom_embeddings": {key: str(path.resolve()) for key, path in embedding_paths.items()},
        "prediction_exports": {key: str(path.resolve()) for key, path in prediction_exports.items()},
        "checkpoint": str(checkpoint_path.resolve()),
        "safetensors": str(safetensors_path.resolve()),
        "metrics": metrics,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Prepared drug table: {prepared_paths['drug_table']}")
    print(f"Prepared protein table: {prepared_paths['protein_table']}")
    print(f"Prepared relation table: {prepared_paths['relation_table']}")
    if exclusions_report is not None:
        print(f"Excluded rows report: {exclusions_report['rows']}")
        print(f"Excluded rows summary: {exclusions_report['summary']}")
    print(f"Predictions export: {prediction_exports['csv']}")
    if "json" in prediction_exports:
        print(f"Predictions export JSON: {prediction_exports['json']}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Safetensors: {safetensors_path}")
    print(f"Custom test AUC: {metrics['test_auc']:.6f}")
    print(f"Custom test AUPRC: {metrics['test_auprc']:.6f}")
    print(f"Custom test F1: {metrics['test_f1']:.6f}")
    print(f"Metrics report: {report_path}")


if __name__ == "__main__":
    main()
