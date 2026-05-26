from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import re
from dataclasses import dataclass
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from datamodule.custom_single_table import SingleTableDTIDataModule
from module.GAT import Net as GATNet
from module.featurizer.drug_featurizer.chembert_featurizer import CHEMFEATURE
from module.featurizer.prot_featurizer.esm_featurizer import ESMFEATURE
from utils import utils


REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_PY = REPO_ROOT / "run.py"


@dataclass(frozen=True)
class ScenarioSpec:
    config_name: str
    best_param_name: str
    config_path: Path
    dataset_name: str
    module_kind: str
    paper_label: str
    overrides: tuple[str, ...] = ()


SCENARIOS = {
    "drugbank": {
        "balanced_warm": ScenarioSpec(
        "drugbank_train_GAT.yaml",
        "random_balanced_GAT.yaml",
        REPO_ROOT / "configs" / "drugbank_train_GAT.yaml",
        "drugbank",
        "GAT",
        ("datamodule.splitting.balanced=True", "datamodule.splitting.splitting_strategy=random", "multiprocessing.multiprocessing=False"),
        ),
        "balanced_cold_drug": ScenarioSpec(
        "drugbank_train_GAT.yaml",
        "cold_drug_balanced_GAT.yaml",
        REPO_ROOT / "configs" / "drugbank_train_GAT.yaml",
        "drugbank",
        "GAT",
        ("datamodule.splitting.balanced=True", "datamodule.splitting.splitting_strategy=cold_drug", "multiprocessing.multiprocessing=False"),
        ),
        "balanced_cold_target": ScenarioSpec(
        "drugbank_train_GAT.yaml",
        "cold_target_balanced.yaml",
        REPO_ROOT / "configs" / "drugbank_train_GAT.yaml",
        "drugbank",
        "GAT",
        ("datamodule.splitting.balanced=True", "datamodule.splitting.splitting_strategy=cold_target", "multiprocessing.multiprocessing=False"),
        ),
        "unbalanced_warm": ScenarioSpec(
        "drugbank_train_GAT.yaml",
        "random_unbalanced.yaml",
        REPO_ROOT / "configs" / "drugbank_train_GAT.yaml",
        "drugbank",
        "GAT",
        ("datamodule.splitting.balanced=False", "datamodule.splitting.splitting_strategy=random", "multiprocessing.multiprocessing=False"),
        ),
        "unbalanced_cold_drug": ScenarioSpec(
        "drugbank_train_GAT.yaml",
        "cold_drug_unbalanced.yaml",
        REPO_ROOT / "configs" / "drugbank_train_GAT.yaml",
        "drugbank",
        "GAT",
        ("datamodule.splitting.balanced=False", "datamodule.splitting.splitting_strategy=cold_drug", "multiprocessing.multiprocessing=False"),
        ),
        "unbalanced_cold_target": ScenarioSpec(
        "drugbank_train_GAT.yaml",
        "cold_target_unbalanced.yaml",
        REPO_ROOT / "configs" / "drugbank_train_GAT.yaml",
        "drugbank",
        "GAT",
        ("datamodule.splitting.balanced=False", "datamodule.splitting.splitting_strategy=cold_target", "multiprocessing.multiprocessing=False"),
        ),
    },
    "bindingDB": {
        "balanced_warm": ScenarioSpec(
        "bindingDB_train_GAT.yaml",
        "random_balanced_GAT.yaml",
        REPO_ROOT / "configs" / "bindingDB_train_GAT.yaml",
        "bindingDB",
        "GAT",
        ("datamodule.splitting.balanced=True", "datamodule.splitting.splitting_strategy=random", "multiprocessing.multiprocessing=False"),
        ),
        "balanced_cold_drug": ScenarioSpec(
        "bindingDB_train_GAT.yaml",
        "cold_drug_balanced_GAT.yaml",
        REPO_ROOT / "configs" / "bindingDB_train_GAT.yaml",
        "bindingDB",
        "GAT",
        ("datamodule.splitting.balanced=True", "datamodule.splitting.splitting_strategy=cold_drug", "multiprocessing.multiprocessing=False"),
        ),
        "balanced_cold_target": ScenarioSpec(
        "bindingDB_train_GAT.yaml",
        "cold_target_balanced.yaml",
        REPO_ROOT / "configs" / "bindingDB_train_GAT.yaml",
        "bindingDB",
        "GAT",
        ("datamodule.splitting.balanced=True", "datamodule.splitting.splitting_strategy=cold_target", "multiprocessing.multiprocessing=False"),
        ),
        "unbalanced_warm": ScenarioSpec(
        "bindingDB_train_GAT.yaml",
        "random_unbalanced.yaml",
        REPO_ROOT / "configs" / "bindingDB_train_GAT.yaml",
        "bindingDB",
        "GAT",
        ("datamodule.splitting.balanced=False", "datamodule.splitting.splitting_strategy=random", "multiprocessing.multiprocessing=False"),
        ),
        "unbalanced_cold_drug": ScenarioSpec(
        "bindingDB_train_GAT.yaml",
        "cold_drug_unbalanced.yaml",
        REPO_ROOT / "configs" / "bindingDB_train_GAT.yaml",
        "bindingDB",
        "GAT",
        ("datamodule.splitting.balanced=False", "datamodule.splitting.splitting_strategy=cold_drug", "multiprocessing.multiprocessing=False"),
        ),
        "unbalanced_cold_target": ScenarioSpec(
        "bindingDB_train_GAT.yaml",
        "cold_target_unbalanced.yaml",
        REPO_ROOT / "configs" / "bindingDB_train_GAT.yaml",
        "bindingDB",
        "GAT",
        ("datamodule.splitting.balanced=False", "datamodule.splitting.splitting_strategy=cold_target", "multiprocessing.multiprocessing=False"),
        ),
    },
    "yamanishi": {
        "warm_start_1_1": ScenarioSpec(
        "yamanishi_train.yaml",
        "yamanishi_GAT.yaml",
        REPO_ROOT / "configs" / "yamanishi_train.yaml",
        "yamanishi",
        "GAT",
        ("preprocess.data_path=data_folds/warm_start_1_1/",),
        ),
        "warm_start_1_10": ScenarioSpec(
        "yamanishi_train.yaml",
        "yamanishi.yaml",
        REPO_ROOT / "configs" / "yamanishi_train.yaml",
        "yamanishi",
        "GAT",
        ("preprocess.data_path=data_folds/warm_start_1_10/",),
        ),
        "drug_coldstart": ScenarioSpec(
        "yamanishi_train.yaml",
        "yamanishi_colddrug.yaml",
        REPO_ROOT / "configs" / "yamanishi_train.yaml",
        "yamanishi",
        "GAT",
        ("preprocess.data_path=data_folds/drug_coldstart/",),
        ),
        "protein_coldstart": ScenarioSpec(
        "yamanishi_train.yaml",
        "yamanishi.yaml",
        REPO_ROOT / "configs" / "yamanishi_train.yaml",
        "yamanishi",
        "GAT",
        ("preprocess.data_path=data_folds/protein_coldstart/",),
        ),
    },
    "luo": {
        "warm_start_1_1": ScenarioSpec(
        "luo_train.yaml",
        "luo_GAT.yaml",
        REPO_ROOT / "configs" / "luo_train.yaml",
        "luo",
        "GAT",
        ("preprocess.data_path=data_folds/warm_start_1_1/",),
        ),
        "warm_start_1_10": ScenarioSpec(
        "luo_train.yaml",
        "luo.yaml",
        REPO_ROOT / "configs" / "luo_train.yaml",
        "luo",
        "GAT",
        ("preprocess.data_path=data_folds/warm_start_1_10/",),
        ),
        "drug_coldstart": ScenarioSpec(
        "luo_train.yaml",
        "luo.yaml",
        REPO_ROOT / "configs" / "luo_train.yaml",
        "luo",
        "GAT",
        ("preprocess.data_path=data_folds/drug_coldstart/",),
        ),
        "protein_coldstart": ScenarioSpec(
        "luo_train.yaml",
        "luo_protcoldstart.yaml",
        REPO_ROOT / "configs" / "luo_train.yaml",
        "luo",
        "GAT",
        ("preprocess.data_path=data_folds/protein_coldstart/",),
        ),
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train on a paper scenario and evaluate on a custom single-table dataset.")
    parser.add_argument("--dataset", required=True, choices=("drugbank", "bindingDB", "yamanishi", "luo"), help="Benchmark dataset to use.")
    parser.add_argument("--scenario", required=True, help="Paper scenario key from reproduce_paper.py.")
    parser.add_argument("--input-csv", required=True, type=Path, help="CSV with Canonical Smiles, Sequence, Activity Value Log, Activity Classification.")
    parser.add_argument("--artifacts-dir", required=True, type=Path, help="Directory for checkpoints and logs.")
    parser.add_argument("--serialized-dir", type=Path, default=REPO_ROOT / "datasets" / "custom_serialized", help="Directory for cached features.")
    parser.add_argument("--checkpoint-name", default="best.ckpt", help="Checkpoint filename to use for the trained model.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU if available.")
    return parser.parse_args()


def normalize_activity(value: str) -> int:
    text = str(value).strip().lower()
    if text in {"active", "gray zone", "gray_zone", "grayzone"}:
        return 1
    if text == "inactive":
        return 0
    raise ValueError(f"Unsupported activity classification: {value}")


def build_pair_table(df: pd.DataFrame):
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


def featurize_unique_entities(data: pd.DataFrame, serialized_dir: Path):
    serialized_dir.mkdir(parents=True, exist_ok=True)
    drug_cache = serialized_dir / "custom_drug_features.pt"
    prot_cache = serialized_dir / "custom_protein_features.pt"

    if drug_cache.exists() and prot_cache.exists():
        X_drug = torch.load(drug_cache)
        X_target = torch.load(prot_cache)
        return X_drug, X_target

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    drug_featurizer = CHEMFEATURE(device)
    prot_featurizer = ESMFEATURE(device)

    unique_drugs = data[["Drug_ID", "SMILES"]].drop_duplicates().sort_values("Drug_ID")
    unique_targets = data[["Prot_ID", "SEQ"]].drop_duplicates().sort_values("Prot_ID")

    drug_features = drug_featurizer.get_representations(unique_drugs["SMILES"].values)
    target_features = prot_featurizer.get_representations(unique_targets["SEQ"].values)

    X_drug = pd.DataFrame(drug_features, index=unique_drugs["Drug_ID"].values)
    X_target = pd.DataFrame(target_features, index=unique_targets["Prot_ID"].values)

    torch.save(X_drug, drug_cache)
    torch.save(X_target, prot_cache)
    return X_drug, X_target


def load_scenario_config(scenario: str):
    spec = SCENARIOS[scenario[0]][scenario[1]]
    return OmegaConf.load(spec.config_path)


def best_param_overrides(best_params_path: Path) -> list[str]:
    if not best_params_path.exists():
        return []

    best_cfg = OmegaConf.to_container(OmegaConf.load(best_params_path), resolve=True)
    overrides = []
    for expr, value in best_cfg.items():
        matches = re.findall(r"\['([^']+)'\]", expr)
        if not matches:
            continue
        overrides.append(f"{'.'.join(matches)}={value!r}" if isinstance(value, str) else f"{'.'.join(matches)}={value}")
    return overrides


def build_model(cfg, dataset):
    module_target = cfg["module"]["_target_"]
    if module_target == "module.GAT.Net":
        return GATNet(cfg, dataset, cfg["module"]["network"], cfg["module"]["optimizer"], cfg["module"]["criterion"], cfg["module"]["GAT_params"])
    raise ValueError(f"Unsupported model architecture: {module_target}")


def scenario_overrides(args, spec: ScenarioSpec):
    artifacts_root = args.artifacts_dir / spec.dataset_name / spec.module_kind / args.dataset / args.scenario
    checkpoint_dir = artifacts_root / "checkpoints"
    hydra_run_dir = artifacts_root / "hydra"
    tb_dir = artifacts_root / "tensorboard"

    overrides = [
        f"callbacks.model_checkpoint.dirpath={checkpoint_dir.as_posix()}/",
        "callbacks.model_checkpoint.save_top_k=1",
        "callbacks.model_checkpoint.save_last=True",
        f"hydra.run.dir={hydra_run_dir.as_posix()}/",
        "hydra.output_subdir=null",
        "hydra.job.chdir=False",
        f"logger.name={spec.dataset_name}",
    ]
    overrides.extend(spec.overrides)
    if spec.dataset_name == "drugbank":
        overrides.append(f"preprocess.data_path={(REPO_ROOT / 'datasets' / 'drugbank').as_posix()}/")
    elif spec.dataset_name == "bindingDB":
        overrides.append(f"preprocess.data_path={(REPO_ROOT / 'datasets' / 'bindingDB').as_posix()}/")
    elif spec.dataset_name == "yamanishi":
        overrides.append(f"preprocess.root_path={(REPO_ROOT / 'datasets' / 'yamanishi_08').as_posix()}/")
    elif spec.dataset_name == "luo":
        luo_root = REPO_ROOT / "datasets" / "luo's_dataset"
        overrides.append(f"preprocess.root_path={luo_root.as_posix()}/")
    overrides.extend(best_param_overrides(REPO_ROOT / "configs" / "best_params" / spec.best_param_name))
    overrides.append(f"datamodule.serializer.save_path={(args.serialized_dir).as_posix()}/")
    overrides.append("datamodule.serializer.load_serialized=True")
    return overrides, artifacts_root, checkpoint_dir, tb_dir


def run_training_scenario(args, spec: ScenarioSpec):
    overrides, artifacts_root, checkpoint_dir, tb_dir = scenario_overrides(args, spec)
    tb_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(RUN_PY),
        "--config-name",
        spec.config_name,
        *overrides,
    ]
    env = dict(os.environ)
    env.setdefault("TENSORBOARD_LOG_DIR", str(tb_dir))
    result = subprocess.run(command, cwd=REPO_ROOT, env=env, check=False, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"Training command failed: {shlex.join(command)}\n{result.stdout}\n{result.stderr}")
    checkpoints = sorted(checkpoint_dir.glob("*.ckpt"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not checkpoints:
        raise RuntimeError(f"No checkpoint found in {checkpoint_dir}")
    return checkpoints[0], artifacts_root


def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    if args.dataset not in SCENARIOS or args.scenario not in SCENARIOS[args.dataset]:
        raise SystemExit(f"Unsupported dataset/scenario combination: {args.dataset}/{args.scenario}")
    spec = SCENARIOS[args.dataset][args.scenario]
    cfg = OmegaConf.to_container(OmegaConf.load(spec.config_path), resolve=True)
    cfg["datamodule"]["serializer"]["load_serialized"] = True

    raw = pd.read_csv(args.input_csv)
    data, pair_table = build_pair_table(raw)
    X_drug, X_target = featurize_unique_entities(data, args.serialized_dir)

    dataset = {
        "X_drug": X_drug,
        "X_target": X_target,
        "test": pair_table.reset_index(drop=True),
    }

    datamodule = SingleTableDTIDataModule(cfg, dataset, cfg["datamodule"]["dm_cfg"], cfg["datamodule"]["splitting"], cfg["datamodule"]["serializer"])
    best_path, artifacts_root = run_training_scenario(args, spec)
    model = build_model(cfg, dataset)
    trained_model = type(model).load_from_checkpoint(
        str(best_path),
        cfg=cfg,
        dataset=dataset,
        network=cfg["module"]["network"],
        optimizer=cfg["module"]["optimizer"],
        criterion=cfg["module"]["criterion"],
        **({"GAT_params": cfg["module"]["GAT_params"]} if "GAT_params" in cfg["module"] else {}),
    )

    trainer = pl.Trainer(accelerator="gpu" if args.use_gpu and torch.cuda.is_available() else "cpu", devices=1, logger=False)
    test_results = trainer.test(trained_model, datamodule=datamodule, ckpt_path=None)
    print(f"Scenario checkpoint: {best_path}")
    print(f"Artifacts: {artifacts_root}")
    print(test_results)


if __name__ == "__main__":
    main()
