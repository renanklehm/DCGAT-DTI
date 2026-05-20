from __future__ import annotations

import argparse
import csv
import os
import platform
import re
import shlex
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_PY = REPO_ROOT / "run.py"


@dataclass(frozen=True)
class Experiment:
    dataset: str
    config_name: str
    scenario_key: str
    paper_label: str
    repeats: int
    best_param_name: str
    overrides: tuple[str, ...] = ()


DRUGBANK_BINDINGDB_EXPERIMENTS = (
    Experiment(
        dataset="drugbank",
        config_name="drugbank_train_GAT.yaml",
        scenario_key="balanced_warm",
        paper_label="Balanced Warm Start",
        repeats=10,
        best_param_name="random_balanced_GAT.yaml",
        overrides=(
            "datamodule.splitting.balanced=True",
            "datamodule.splitting.splitting_strategy=random",
            "multiprocessing.multiprocessing=False",
        ),
    ),
    Experiment(
        dataset="drugbank",
        config_name="drugbank_train_GAT.yaml",
        scenario_key="balanced_cold_drug",
        paper_label="Balanced Cold Start for Drug",
        repeats=10,
        best_param_name="cold_drug_balanced_GAT.yaml",
        overrides=(
            "datamodule.splitting.balanced=True",
            "datamodule.splitting.splitting_strategy=cold_drug",
            "multiprocessing.multiprocessing=False",
        ),
    ),
    Experiment(
        dataset="drugbank",
        config_name="drugbank_train_GAT.yaml",
        scenario_key="balanced_cold_target",
        paper_label="Balanced Cold Start for Protein",
        repeats=10,
        best_param_name="cold_target_balanced.yaml",
        overrides=(
            "datamodule.splitting.balanced=True",
            "datamodule.splitting.splitting_strategy=cold_target",
            "multiprocessing.multiprocessing=False",
        ),
    ),
    Experiment(
        dataset="drugbank",
        config_name="drugbank_train_GAT.yaml",
        scenario_key="unbalanced_warm",
        paper_label="Unbalanced Warm Start",
        repeats=10,
        best_param_name="random_unbalanced.yaml",
        overrides=(
            "datamodule.splitting.balanced=False",
            "datamodule.splitting.splitting_strategy=random",
            "multiprocessing.multiprocessing=False",
        ),
    ),
    Experiment(
        dataset="drugbank",
        config_name="drugbank_train_GAT.yaml",
        scenario_key="unbalanced_cold_drug",
        paper_label="Unbalanced Cold Start for Drug",
        repeats=10,
        best_param_name="cold_drug_unbalanced.yaml",
        overrides=(
            "datamodule.splitting.balanced=False",
            "datamodule.splitting.splitting_strategy=cold_drug",
            "multiprocessing.multiprocessing=False",
        ),
    ),
    Experiment(
        dataset="drugbank",
        config_name="drugbank_train_GAT.yaml",
        scenario_key="unbalanced_cold_target",
        paper_label="Unbalanced Cold Start for Protein",
        repeats=10,
        best_param_name="cold_target_unbalanced.yaml",
        overrides=(
            "datamodule.splitting.balanced=False",
            "datamodule.splitting.splitting_strategy=cold_target",
            "multiprocessing.multiprocessing=False",
        ),
    ),
    Experiment(
        dataset="bindingDB",
        config_name="bindingDB_train_GAT.yaml",
        scenario_key="balanced_warm",
        paper_label="Balanced Warm Start",
        repeats=10,
        best_param_name="random_balanced_GAT.yaml",
        overrides=(
            "datamodule.splitting.balanced=True",
            "datamodule.splitting.splitting_strategy=random",
            "multiprocessing.multiprocessing=False",
        ),
    ),
    Experiment(
        dataset="bindingDB",
        config_name="bindingDB_train_GAT.yaml",
        scenario_key="balanced_cold_drug",
        paper_label="Balanced Cold Start for Drug",
        repeats=10,
        best_param_name="cold_drug_balanced_GAT.yaml",
        overrides=(
            "datamodule.splitting.balanced=True",
            "datamodule.splitting.splitting_strategy=cold_drug",
            "multiprocessing.multiprocessing=False",
        ),
    ),
    Experiment(
        dataset="bindingDB",
        config_name="bindingDB_train_GAT.yaml",
        scenario_key="balanced_cold_target",
        paper_label="Balanced Cold Start for Protein",
        repeats=10,
        best_param_name="cold_target_balanced.yaml",
        overrides=(
            "datamodule.splitting.balanced=True",
            "datamodule.splitting.splitting_strategy=cold_target",
            "multiprocessing.multiprocessing=False",
        ),
    ),
    Experiment(
        dataset="bindingDB",
        config_name="bindingDB_train_GAT.yaml",
        scenario_key="unbalanced_warm",
        paper_label="Unbalanced Warm Start",
        repeats=10,
        best_param_name="random_unbalanced.yaml",
        overrides=(
            "datamodule.splitting.balanced=False",
            "datamodule.splitting.splitting_strategy=random",
            "multiprocessing.multiprocessing=False",
        ),
    ),
    Experiment(
        dataset="bindingDB",
        config_name="bindingDB_train_GAT.yaml",
        scenario_key="unbalanced_cold_drug",
        paper_label="Unbalanced Cold Start for Drug",
        repeats=10,
        best_param_name="cold_drug_unbalanced.yaml",
        overrides=(
            "datamodule.splitting.balanced=False",
            "datamodule.splitting.splitting_strategy=cold_drug",
            "multiprocessing.multiprocessing=False",
        ),
    ),
    Experiment(
        dataset="bindingDB",
        config_name="bindingDB_train_GAT.yaml",
        scenario_key="unbalanced_cold_target",
        paper_label="Unbalanced Cold Start for Protein",
        repeats=10,
        best_param_name="cold_target_unbalanced.yaml",
        overrides=(
            "datamodule.splitting.balanced=False",
            "datamodule.splitting.splitting_strategy=cold_target",
            "multiprocessing.multiprocessing=False",
        ),
    ),
)


# These two datasets are provided as 10 predefined folds by the upstream source used in the paper.
# The repository's preprocessors already iterate across all 10 folds, so each scenario is one run.
FOLD_EXPERIMENTS = (
    Experiment(
        dataset="yamanishi",
        config_name="yamanishi_train.yaml",
        scenario_key="warm_start_1_1",
        paper_label="Balanced Warm Start",
        repeats=1,
        best_param_name="yamanishi_GAT.yaml",
        overrides=("preprocess.data_path=data_folds/warm_start_1_1/",),
    ),
    Experiment(
        dataset="yamanishi",
        config_name="yamanishi_train.yaml",
        scenario_key="warm_start_1_10",
        paper_label="Unbalanced Warm Start",
        repeats=1,
        best_param_name="yamanishi.yaml",
        overrides=("preprocess.data_path=data_folds/warm_start_1_10/",),
    ),
    Experiment(
        dataset="yamanishi",
        config_name="yamanishi_train.yaml",
        scenario_key="drug_coldstart",
        paper_label="Cold Start for Drug",
        repeats=1,
        best_param_name="yamanishi_colddrug.yaml",
        overrides=("preprocess.data_path=data_folds/drug_coldstart/",),
    ),
    Experiment(
        dataset="yamanishi",
        config_name="yamanishi_train.yaml",
        scenario_key="protein_coldstart",
        paper_label="Cold Start for Protein",
        repeats=1,
        best_param_name="yamanishi.yaml",
        overrides=("preprocess.data_path=data_folds/protein_coldstart/",),
    ),
    Experiment(
        dataset="luo",
        config_name="luo_train.yaml",
        scenario_key="warm_start_1_1",
        paper_label="Balanced Warm Start",
        repeats=1,
        best_param_name="luo_GAT.yaml",
        overrides=("preprocess.data_path=data_folds/warm_start_1_1/",),
    ),
    Experiment(
        dataset="luo",
        config_name="luo_train.yaml",
        scenario_key="warm_start_1_10",
        paper_label="Unbalanced Warm Start",
        repeats=1,
        best_param_name="luo.yaml",
        overrides=("preprocess.data_path=data_folds/warm_start_1_10/",),
    ),
    Experiment(
        dataset="luo",
        config_name="luo_train.yaml",
        scenario_key="drug_coldstart",
        paper_label="Cold Start for Drug",
        repeats=1,
        best_param_name="luo.yaml",
        overrides=("preprocess.data_path=data_folds/drug_coldstart/",),
    ),
    Experiment(
        dataset="luo",
        config_name="luo_train.yaml",
        scenario_key="protein_coldstart",
        paper_label="Cold Start for Protein",
        repeats=1,
        best_param_name="luo_protcoldstart.yaml",
        overrides=("preprocess.data_path=data_folds/protein_coldstart/",),
    ),
)


ALL_EXPERIMENTS = DRUGBANK_BINDINGDB_EXPERIMENTS + FOLD_EXPERIMENTS

SERIALIZED_FILES = {
    "drugbank": ("DrugBank_PubChem10M.pt", "DrugBank_ESM.pt"),
    "bindingDB": ("bindingDB_Kd_PubChem10M.pt", "bindingDB_Kd_ESM.pt"),
    "yamanishi": ("yamanishi_PubChem10M.pt", "yamanishi_ESM.pt"),
    "luo": ("luo_PubChem10M.pt", "luo_ESM.pt"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce the paper experiments on Linux with a CUDA-capable GPU.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("all", "drugbank", "bindingDB", "yamanishi", "luo"),
        default=["all"],
        help="Subset of datasets to run. Defaults to all.",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=42,
        help="First split seed for DrugBank and BindingDB random/cold-start repeats.",
    )
    parser.add_argument(
        "--seed-count",
        type=int,
        default=10,
        help="Number of repeated split seeds for DrugBank and BindingDB.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "paper_reproduction",
        help="Directory for checkpoints, captured logs, Hydra outputs, and summary files.",
    )
    parser.add_argument(
        "--serialized-dir",
        type=Path,
        default=REPO_ROOT / "datasets" / "serialized",
        help="Directory containing or receiving serialized drug/protein embeddings.",
    )
    parser.add_argument(
        "--yamanishi-root",
        type=Path,
        default=REPO_ROOT / "datasets" / "yamanishi_08",
        help="Root directory containing the Yamanishi fold and feature files.",
    )
    parser.add_argument(
        "--luo-root",
        type=Path,
        default=REPO_ROOT / "datasets" / "luo_dataset",
        help="Root directory containing the Luo fold, feature, and mapping files.",
    )
    parser.add_argument(
        "--bindingdb-root",
        type=Path,
        default=REPO_ROOT / "datasets" / "bindingDB",
        help="BindingDB dataset directory used by preprocess.bindingDB.data_path.",
    )
    parser.add_argument(
        "--drugbank-root",
        type=Path,
        default=REPO_ROOT / "datasets" / "drugbank",
        help="DrugBank dataset directory used by preprocess.drugbank.data_path.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a run when its captured log file already exists.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only create serialized embeddings, do not launch training runs.",
    )
    parser.add_argument(
        "--skip-env-checks",
        action="store_true",
        help="Skip the Linux and GPU availability checks.",
    )
    return parser.parse_args()


def ensure_supported_environment(skip_checks: bool) -> None:
    if skip_checks:
        return

    if platform.system() != "Linux":
        raise SystemExit("This reproduction script is intended for Linux only.")

    try:
        subprocess.run(
            ["nvidia-smi"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as exc:  # pragma: no cover - defensive CLI guard
        raise SystemExit("A visible NVIDIA GPU is required for these experiments.") from exc


def normalize_dir(path: Path) -> str:
    return path.resolve().as_posix().rstrip("/") + "/"


def hydra_override(value: str) -> str:
    return value


def base_overrides(args: argparse.Namespace, experiment: Experiment, run_tag: str) -> list[str]:
    checkpoints_dir = args.artifacts_dir / "checkpoints" / experiment.dataset / experiment.scenario_key / run_tag
    tensorboard_dir = args.artifacts_dir / "tensorboard"

    overrides = [
        hydra_override("tuning.param_search.tune=False"),
        hydra_override(f"best_param_name={experiment.best_param_name}"),
        hydra_override(f"callbacks.model_checkpoint.dirpath={normalize_dir(checkpoints_dir)}"),
        hydra_override("callbacks.model_checkpoint.save_top_k=0"),
        hydra_override("callbacks.model_checkpoint.save_last=False"),
        hydra_override(f"datamodule.serializer.save_path={normalize_dir(args.serialized_dir)}"),
        hydra_override(f"logger.name={experiment.dataset}"),
        hydra_override(f"hydra.run.dir={normalize_dir(args.artifacts_dir / 'hydra_runs' / experiment.dataset / experiment.scenario_key / run_tag)}"),
        hydra_override("hydra.output_subdir=null"),
        hydra_override("hydra.job.chdir=False"),
    ]

    # Keep the hardcoded TensorBoard path inside the repo tree.
    os.environ.setdefault("TENSORBOARD_LOG_DIR", str(tensorboard_dir))

    if experiment.dataset == "drugbank":
        overrides.append(hydra_override(f"preprocess.data_path={normalize_dir(args.drugbank_root)}"))
    elif experiment.dataset == "bindingDB":
        overrides.append(hydra_override(f"preprocess.data_path={normalize_dir(args.bindingdb_root)}"))
    elif experiment.dataset == "yamanishi":
        overrides.append(hydra_override(f"preprocess.root_path={normalize_dir(args.yamanishi_root)}"))
        overrides.append(hydra_override("multiprocessing.multiprocessing=True"))
        overrides.append(hydra_override("multiprocessing.num_process=10"))
        overrides.append(hydra_override("multiprocessing.concurrent_process=1"))
    elif experiment.dataset == "luo":
        overrides.append(hydra_override(f"preprocess.root_path={normalize_dir(args.luo_root)}"))
        overrides.append(hydra_override("multiprocessing.multiprocessing=True"))
        overrides.append(hydra_override("multiprocessing.num_process=10"))
        overrides.append(hydra_override("multiprocessing.concurrent_process=1"))

    overrides.extend(experiment.overrides)
    return overrides


def run_command(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("COMMAND: " + shlex.join(command) + "\n\n")
        log_file.flush()
        process = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    if process.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {process.returncode}: {shlex.join(command)}")


def parse_single_run_metrics(log_text: str) -> tuple[float, float]:
    auc_matches = re.findall(r"Test AUC:\s*([0-9]*\.?[0-9]+)", log_text)
    auprc_matches = re.findall(r"Test AUPRC:\s*([0-9]*\.?[0-9]+)", log_text)
    if not auc_matches or not auprc_matches:
        raise ValueError("Could not find Test AUC/Test AUPRC in run log.")
    return float(auc_matches[-1]), float(auprc_matches[-1])


def parse_mean_metrics(log_text: str) -> tuple[float, float]:
    auc_matches = re.findall(r"Mean test AUC:\s*([0-9]*\.?[0-9]+)", log_text)
    auprc_matches = re.findall(r"Mean test AUPRC:\s*([0-9]*\.?[0-9]+)", log_text)
    if not auc_matches or not auprc_matches:
        raise ValueError("Could not find Mean test AUC/Mean test AUPRC in run log.")
    return float(auc_matches[-1]), float(auprc_matches[-1])


def parse_metrics_for_experiment(experiment: Experiment, log_text: str) -> tuple[float, float]:
    if experiment.dataset in {"drugbank", "bindingDB"}:
        return parse_single_run_metrics(log_text)
    return parse_mean_metrics(log_text)


def ensure_serialized_features(args: argparse.Namespace, dataset: str) -> None:
    expected_files = [args.serialized_dir / name for name in SERIALIZED_FILES[dataset]]
    if all(path.exists() for path in expected_files):
        return

    config_by_dataset = {
        "drugbank": "drugbank_train_GAT.yaml",
        "bindingDB": "bindingDB_train_GAT.yaml",
        "yamanishi": "yamanishi_train.yaml",
        "luo": "luo_train.yaml",
    }
    bootstrap_experiment = Experiment(
        dataset=dataset,
        config_name=config_by_dataset[dataset],
        scenario_key="feature_bootstrap",
        paper_label="Feature Bootstrap",
        repeats=1,
        best_param_name="random_balanced_GAT.yaml",
    )
    log_path = args.artifacts_dir / "bootstrap_logs" / f"{dataset}.log"
    overrides = base_overrides(args, bootstrap_experiment, "bootstrap")
    overrides.append("datamodule.serializer.load_serialized=False")
    overrides.append("multiprocessing.multiprocessing=False")

    command = [sys.executable, str(RUN_PY), "--config-name", bootstrap_experiment.config_name, *overrides]
    run_command(command, log_path)

    missing = [str(path) for path in expected_files if not path.exists()]
    if missing:
        raise RuntimeError(f"Expected serialized features were not created: {missing}")


def selected_experiments(args: argparse.Namespace) -> list[Experiment]:
    if "all" in args.datasets:
        selected = {"drugbank", "bindingDB", "yamanishi", "luo"}
    else:
        selected = set(args.datasets)
    return [experiment for experiment in ALL_EXPERIMENTS if experiment.dataset in selected]


def summarize(values: list[float]) -> tuple[float, float]:
    mean_value = statistics.mean(values)
    std_value = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean_value, std_value


def main() -> None:
    args = parse_args()
    ensure_supported_environment(args.skip_env_checks)

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    args.serialized_dir.mkdir(parents=True, exist_ok=True)

    experiments = selected_experiments(args)
    for dataset in sorted({experiment.dataset for experiment in experiments}):
        ensure_serialized_features(args, dataset)

    if args.prepare_only:
        print("Serialized features are ready.")
        return

    summary_rows: list[dict[str, str | float | int]] = []
    repeated_seeds = [args.seed_start + offset for offset in range(args.seed_count)]

    for experiment in experiments:
        auc_values: list[float] = []
        auprc_values: list[float] = []

        if experiment.dataset in {"drugbank", "bindingDB"}:
            run_tags = [f"seed_{seed}" for seed in repeated_seeds]
            run_overrides = [
                (tag, [f"datamodule.splitting.seed={seed}"])
                for tag, seed in zip(run_tags, repeated_seeds, strict=True)
            ]
        else:
            run_overrides = [("folds_1_to_10", [])]

        for run_tag, extra_overrides in run_overrides:
            log_path = args.artifacts_dir / "run_logs" / experiment.dataset / experiment.scenario_key / f"{run_tag}.log"
            log_text = None
            if args.skip_existing and log_path.exists():
                cached_log_text = log_path.read_text(encoding="utf-8")
                try:
                    parse_metrics_for_experiment(experiment, cached_log_text)
                except ValueError:
                    print(f"Re-running {experiment.dataset}/{experiment.scenario_key}/{run_tag} because cached log has no final metrics.")
                else:
                    log_text = cached_log_text

            if log_text is None:
                overrides = base_overrides(args, experiment, run_tag)
                overrides.extend(extra_overrides)
                command = [sys.executable, str(RUN_PY), "--config-name", experiment.config_name, *overrides]
                run_command(command, log_path)
                log_text = log_path.read_text(encoding="utf-8")

            auc, auprc = parse_metrics_for_experiment(experiment, log_text)

            auc_values.append(auc)
            auprc_values.append(auprc)

        auc_mean, auc_std = summarize(auc_values)
        auprc_mean, auprc_std = summarize(auprc_values)
        summary_rows.append(
            {
                "dataset": experiment.dataset,
                "scenario": experiment.paper_label,
                "scenario_key": experiment.scenario_key,
                "runs": len(auc_values),
                "mean_auc": round(auc_mean, 6),
                "std_auc": round(auc_std, 6),
                "mean_auprc": round(auprc_mean, 6),
                "std_auprc": round(auprc_std, 6),
                "best_param_name": experiment.best_param_name,
            }
        )

    summary_path = args.artifacts_dir / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "dataset",
                "scenario",
                "scenario_key",
                "runs",
                "mean_auc",
                "std_auc",
                "mean_auprc",
                "std_auprc",
                "best_param_name",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
