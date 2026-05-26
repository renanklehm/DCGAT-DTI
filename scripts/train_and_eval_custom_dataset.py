from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir

from datamodule.custom_single_table import SingleTableDTIDataModule
from reproduce_paper import ALL_EXPERIMENTS, base_overrides, ensure_serialized_features, normalize_dir, run_command
from utils import utils


RUN_PY = REPO_ROOT / "run.py"


def parse_args():
    parser = argparse.ArgumentParser(description="Train on a paper scenario and evaluate on a custom single-table dataset.")
    parser.add_argument("--dataset", required=True, choices=("drugbank", "bindingDB", "yamanishi", "luo"), help="Benchmark dataset to use.")
    parser.add_argument("--scenario", required=True, help="Paper scenario key from reproduce_paper.py.")
    parser.add_argument("--input-csv", required=True, type=Path, help="CSV or JSON with Canonical Smiles, Sequence, Activity Value Log, Activity Classification.")
    parser.add_argument("--artifacts-dir", required=True, type=Path, help="Directory for checkpoints and logs.")
    parser.add_argument("--benchmark-serialized-dir", type=Path, default=REPO_ROOT / "datasets" / "serialized", help="Directory containing serialized benchmark features used for scenario training.")
    parser.add_argument("--custom-serialized-dir", type=Path, default=REPO_ROOT / "datasets" / "custom_serialized", help="Directory for cached features of the unseen evaluation dataset.")
    parser.add_argument("--drugbank-root", type=Path, default=REPO_ROOT / "datasets" / "drugbank", help="DrugBank dataset directory used by preprocess.drugbank.data_path.")
    parser.add_argument("--bindingdb-root", type=Path, default=REPO_ROOT / "datasets" / "bindingDB", help="BindingDB dataset directory used by preprocess.bindingDB.data_path.")
    parser.add_argument("--yamanishi-root", type=Path, default=REPO_ROOT / "datasets" / "yamanishi_08", help="Root directory containing the Yamanishi fold and feature files.")
    parser.add_argument("--luo-root", type=Path, default=REPO_ROOT / "datasets" / "luo's_dataset", help="Root directory containing the Luo fold, feature, and mapping files.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU if available.")
    return parser.parse_args()


def load_input_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return pd.read_json(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format: {path.suffix}")


def normalize_activity(value: str) -> int:
    text = str(value).strip().lower()
    if text in {"active", "gray zone", "gray_zone", "grayzone"}:
        return 1
    if text == "inactive":
        return 0
    raise ValueError(f"Unsupported activity classification: {value}")


def build_pair_table(df: pd.DataFrame):
    required_columns = {
        "Canonical Smiles",
        "Sequence",
        "Activity Value Log",
        "Activity Classification",
    }
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    data = df.copy()
    data = data.rename(columns={
        "Canonical Smiles": "SMILES",
        "Sequence": "SEQ",
        "Activity Value Log": "activity_value_log",
        "Activity Classification": "activity_classification",
    })
    data["label"] = data["activity_classification"].map(normalize_activity)
    data["Drug_ID"] = data["SMILES"].astype(str).map({v: i for i, v in enumerate(data["SMILES"].drop_duplicates())})
    data["Prot_ID"] = data["SEQ"].astype(str).map({v: i for i, v in enumerate(data["SEQ"].drop_duplicates())})
    pairs = data[["Drug_ID", "Prot_ID", "label"]].copy()
    pairs["Drug_ID"] = pairs["Drug_ID"].astype(int)
    pairs["Prot_ID"] = pairs["Prot_ID"].astype(int)
    return data, pairs


def featurize_unique_entities(cfg, data: pd.DataFrame, serialized_dir: Path):
    serialized_dir.mkdir(parents=True, exist_ok=True)
    drug_cache = serialized_dir / cfg["datamodule"]["serializer"]["drug_name"]
    prot_cache = serialized_dir / cfg["datamodule"]["serializer"]["target_name"]

    if drug_cache.exists() and prot_cache.exists():
        X_drug = torch.load(drug_cache)
        X_target = torch.load(prot_cache)
        return X_drug, X_target

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    drug_featurizer = hydra.utils.instantiate(cfg["featurizer"]["drugfeaturizer"], device, _recursive_=False)
    prot_featurizer = hydra.utils.instantiate(cfg["featurizer"]["protfeaturizer"], device, _recursive_=False)

    unique_drugs = data[["Drug_ID", "SMILES"]].drop_duplicates().sort_values("Drug_ID")
    unique_targets = data[["Prot_ID", "SEQ"]].drop_duplicates().sort_values("Prot_ID")

    drug_features = drug_featurizer.get_representations(unique_drugs["SMILES"].values)
    target_features = prot_featurizer.get_representations(unique_targets["SEQ"].values)

    X_drug = pd.DataFrame(drug_features, index=unique_drugs["Drug_ID"].values)
    X_target = pd.DataFrame(target_features, index=unique_targets["Prot_ID"].values)

    torch.save(X_drug, drug_cache)
    torch.save(X_target, prot_cache)
    return X_drug, X_target


def find_experiment(dataset: str, scenario: str):
    for experiment in ALL_EXPERIMENTS:
        if experiment.dataset == dataset and experiment.scenario_key == scenario:
            return experiment
    raise SystemExit(f"Unsupported dataset/scenario combination: {dataset}/{scenario}")


def set_override(overrides: list[str], key: str, value: str) -> None:
    prefix = f"{key}="
    for index, override in enumerate(overrides):
        if override.startswith(prefix):
            overrides[index] = f"{prefix}{value}"
            return
    overrides.append(f"{prefix}{value}")


def make_reproduction_args(args) -> SimpleNamespace:
    return SimpleNamespace(
        artifacts_dir=args.artifacts_dir,
        serialized_dir=args.benchmark_serialized_dir,
        yamanishi_root=args.yamanishi_root,
        luo_root=args.luo_root,
        bindingdb_root=args.bindingdb_root,
        drugbank_root=args.drugbank_root,
    )


def scenario_overrides(args, experiment):
    run_tag = f"custom_eval_seed_{args.seed}"
    artifacts_root = args.artifacts_dir / experiment.dataset / experiment.scenario_key / run_tag
    checkpoint_dir = artifacts_root / "checkpoints"
    tensorboard_dir = artifacts_root / "tensorboard"

    overrides = base_overrides(make_reproduction_args(args), experiment, run_tag)
    set_override(overrides, "callbacks.model_checkpoint.dirpath", normalize_dir(checkpoint_dir))
    set_override(overrides, "callbacks.model_checkpoint.save_top_k", "1")
    set_override(overrides, "callbacks.model_checkpoint.save_last", "True")

    if experiment.dataset in {"drugbank", "bindingDB"}:
        set_override(overrides, "datamodule.splitting.seed", str(args.seed))

    return overrides, artifacts_root, checkpoint_dir, tensorboard_dir


def load_scenario_config(experiment, overrides: list[str]):
    with initialize_config_dir(version_base="1.3", config_dir=str(REPO_ROOT / "configs")):
        cfg = compose(config_name=experiment.config_name, overrides=overrides)

    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg["best_param_name"] = experiment.best_param_name
    return utils.update_best_param(cfg)


def load_trained_model(cfg, dataset, checkpoint_path: Path):
    model_class = hydra.utils.get_class(cfg["module"]["_target_"])
    checkpoint_kwargs = {
        "cfg": cfg,
        "dataset": dataset,
        "network": cfg["module"]["network"],
        "optimizer": cfg["module"]["optimizer"],
        "criterion": cfg["module"]["criterion"],
    }
    if "GAT_params" in cfg["module"]:
        checkpoint_kwargs["GAT_params"] = cfg["module"]["GAT_params"]
    return model_class.load_from_checkpoint(str(checkpoint_path), **checkpoint_kwargs)


def run_training_scenario(args, experiment):
    overrides, artifacts_root, checkpoint_dir, tensorboard_dir = scenario_overrides(args, experiment)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TENSORBOARD_LOG_DIR", str(tensorboard_dir))

    log_path = artifacts_root / "training.log"
    command = [sys.executable, str(RUN_PY), "--config-name", experiment.config_name, *overrides]
    run_command(command, log_path)

    checkpoints = sorted(checkpoint_dir.glob("*.ckpt"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not checkpoints:
        raise RuntimeError(f"No checkpoint found in {checkpoint_dir}")
    return checkpoints[0], artifacts_root, overrides


def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    experiment = find_experiment(args.dataset, args.scenario)

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    args.benchmark_serialized_dir.mkdir(parents=True, exist_ok=True)
    args.custom_serialized_dir.mkdir(parents=True, exist_ok=True)

    ensure_serialized_features(make_reproduction_args(args), args.dataset)
    best_path, artifacts_root, overrides = run_training_scenario(args, experiment)
    cfg = load_scenario_config(experiment, overrides)

    raw = load_input_table(args.input_csv)
    data, pair_table = build_pair_table(raw)
    X_drug, X_target = featurize_unique_entities(cfg, data, args.custom_serialized_dir)

    dataset = {
        "X_drug": X_drug,
        "X_target": X_target,
        "test": pair_table.reset_index(drop=True),
    }

    datamodule = SingleTableDTIDataModule(cfg, dataset, cfg["datamodule"]["dm_cfg"], cfg["datamodule"]["splitting"], cfg["datamodule"]["serializer"])
    trained_model = load_trained_model(cfg, dataset, best_path)

    trainer = pl.Trainer(accelerator="gpu" if args.use_gpu and torch.cuda.is_available() else "cpu", devices=1, logger=False)
    test_results = trainer.test(trained_model, datamodule=datamodule, ckpt_path=None)
    print(f"Scenario checkpoint: {best_path}")
    print(f"Artifacts: {artifacts_root}")
    print(test_results)


if __name__ == "__main__":
    main()
