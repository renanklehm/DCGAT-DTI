from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_PY = REPO_ROOT / "run.py"
REPRODUCE_PAPER_PY = REPO_ROOT / "scripts" / "reproduce_paper.py"
DEFAULT_SERIALIZED_DIR = REPO_ROOT / "datasets" / "serialized"
DEFAULT_ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "custom_eval"


@dataclass(frozen=True)
class CustomDatasetTables:
    X_drug: pd.DataFrame
    X_target: pd.DataFrame
    DTI: pd.DataFrame


@dataclass(frozen=True)
class FilteredCustomData:
    frame: pd.DataFrame
    excluded: pd.DataFrame
    original: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train one existing paper scenario on a built-in dataset, export the trained "
            "weights to safetensors, and evaluate them on a custom smiles|sequence|activation file."
        )
    )
    parser.add_argument(
        "--custom-data",
        type=Path,
        required=True,
        help="Path to the custom dataset file in smiles|sequence|activation format.",
    )
    parser.add_argument(
        "--scenario",
        required=True,
        help=(
            "Scenario to train, in dataset:scenario_key form. "
            "Example: drugbank:balanced_warm or bindingDB:unbalanced_cold_target."
        ),
    )
    parser.add_argument(
        "--delimiter",
        default="|",
        help="Field delimiter for the custom dataset file. Defaults to '|'.",
    )
    parser.add_argument(
        "--has-header",
        action="store_true",
        help="Treat the first row of the custom dataset file as a header row.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Split seed for DrugBank/BindingDB scenarios. Defaults to 42.",
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
    return parser.parse_args()


def normalize_dir(path: Path) -> str:
    return path.resolve().as_posix().rstrip("/") + "/"


def strip_yaml_suffix(config_name: str) -> str:
    return config_name[:-5] if config_name.endswith(".yaml") else config_name


def sanitize_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return slug.strip("._-") or "custom"


def load_reproduce_paper_module() -> Any:
    spec = importlib.util.spec_from_file_location("reproduce_paper", REPRODUCE_PAPER_PY)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {REPRODUCE_PAPER_PY}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def get_experiment(module: Any, scenario: str) -> Any:
    if ":" not in scenario:
        raise ValueError("Scenario must be provided as dataset:scenario_key.")
    dataset, scenario_key = scenario.split(":", 1)
    for experiment in module.ALL_EXPERIMENTS:
        if experiment.dataset == dataset and experiment.scenario_key == scenario_key:
            return experiment
    available = sorted(f"{exp.dataset}:{exp.scenario_key}" for exp in module.ALL_EXPERIMENTS)
    raise ValueError(f"Unknown scenario '{scenario}'. Available scenarios: {', '.join(available)}")


def read_custom_triplets(path: Path, delimiter: str, has_header: bool) -> FilteredCustomData:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path, sep=delimiter, header=0 if has_header else None)
    elif suffix == ".json":
        frame = pd.read_json(path)
    else:
        raise ValueError("Custom dataset must be a .csv or .json file.")

    if frame.empty:
        raise ValueError(f"No data rows were found in {path}")

    if suffix == ".json":
        expected_columns = {"smiles", "sequence", "activation"}
        lowered = {str(column).strip().lower(): column for column in frame.columns}
        if not expected_columns.issubset(lowered):
            raise ValueError(
                "JSON input must contain columns named smiles, sequence, and activation."
            )
        frame = frame.rename(
            columns={
                lowered["smiles"]: "SMILES",
                lowered["sequence"]: "SEQ",
                lowered["activation"]: "label",
            }
        )
        frame = frame[["SMILES", "SEQ", "label"]]
    else:
        if has_header:
            lowered = {str(column).strip().lower(): column for column in frame.columns}
            expected_columns = {"smiles", "sequence", "activation"}
            if not expected_columns.issubset(lowered):
                raise ValueError(
                    "CSV input with --has-header must contain columns named smiles, sequence, and activation."
                )
            frame = frame.rename(
                columns={
                    lowered["smiles"]: "SMILES",
                    lowered["sequence"]: "SEQ",
                    lowered["activation"]: "label",
                }
            )
            frame = frame[["SMILES", "SEQ", "label"]]
        else:
            if frame.shape[1] != 3:
                raise ValueError(f"Expected 3 columns in {path}, found {frame.shape[1]}")
            frame = frame.iloc[:, :3]
            frame.columns = ["SMILES", "SEQ", "label"]

    frame["SMILES"] = frame["SMILES"].astype(str).str.strip()
    frame["SEQ"] = frame["SEQ"].astype(str).str.strip()
    frame["label"] = frame["label"].astype(str).str.strip()
    frame = frame.reset_index(drop=True)
    frame.insert(0, "source_row", frame.index.astype(int))

    invalid_labels = ~frame["label"].isin({"0", "1"})
    invalid_sequences = ~frame["SEQ"].str.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+")
    too_long_smiles = frame["SMILES"].str.len() > 510
    too_long_sequences = frame["SEQ"].str.len() > 700
    empty_smiles = frame["SMILES"].eq("")
    empty_sequences = frame["SEQ"].eq("")

    exclusion_reason = pd.Series("", index=frame.index, dtype="object")
    exclusion_reason = exclusion_reason.mask(empty_smiles, exclusion_reason.where(~empty_smiles, "empty_smiles"))
    exclusion_reason = exclusion_reason.mask(
        empty_sequences & exclusion_reason.eq(""),
        exclusion_reason.where(~(empty_sequences & exclusion_reason.eq("")), "empty_sequence"),
    )
    exclusion_reason = exclusion_reason.mask(
        invalid_labels & exclusion_reason.eq(""),
        exclusion_reason.where(~(invalid_labels & exclusion_reason.eq("")), "invalid_activation"),
    )
    exclusion_reason = exclusion_reason.mask(
        invalid_sequences & exclusion_reason.eq(""),
        exclusion_reason.where(~(invalid_sequences & exclusion_reason.eq("")), "non_canonical_sequence"),
    )
    exclusion_reason = exclusion_reason.mask(
        too_long_smiles & exclusion_reason.eq(""),
        exclusion_reason.where(~(too_long_smiles & exclusion_reason.eq("")), "smiles_too_long"),
    )
    exclusion_reason = exclusion_reason.mask(
        too_long_sequences & exclusion_reason.eq(""),
        exclusion_reason.where(~(too_long_sequences & exclusion_reason.eq("")), "sequence_too_long"),
    )

    excluded_mask = exclusion_reason.ne("")
    excluded = frame.loc[excluded_mask].copy()
    if not excluded.empty:
        excluded.insert(1, "reason", exclusion_reason.loc[excluded_mask].values)

    filtered = frame.loc[~excluded_mask].copy()
    if filtered.empty:
        raise ValueError("All custom rows were filtered out. Check the exclusions report for details.")

    filtered["label"] = filtered["label"].astype(int)

    return FilteredCustomData(frame=filtered, excluded=excluded, original=frame.copy())


def save_exclusions_report(excluded: pd.DataFrame, output_dir: Path) -> dict[str, Path] | None:
    if excluded.empty:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    exclusions_path = output_dir / "excluded_rows.tsv"
    summary_path = output_dir / "excluded_summary.json"
    excluded.to_csv(exclusions_path, sep="\t", index=False)
    summary = {
        "excluded_rows": int(len(excluded)),
        "reasons": excluded["reason"].value_counts().sort_index().to_dict(),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"rows": exclusions_path, "summary": summary_path}


def save_prediction_export(
    filtered_custom_data: FilteredCustomData,
    prediction_rows: pd.DataFrame,
    output_dir: Path,
    input_path: Path,
    delimiter: str,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    export_frame = filtered_custom_data.original.copy()
    export_frame = export_frame.rename(columns={"SMILES": "smiles", "SEQ": "sequence", "label": "activity"})
    export_frame["excluded_reason"] = ""
    export_frame["predicted_label"] = pd.Series([pd.NA] * len(export_frame), dtype="Int64")
    export_frame["probability_active"] = np.nan
    export_frame["probability_inactive"] = np.nan

    if not filtered_custom_data.excluded.empty:
        exclusion_reasons = filtered_custom_data.excluded.set_index("source_row")["reason"]
        export_frame.loc[export_frame["source_row"].isin(exclusion_reasons.index), "excluded_reason"] = (
            export_frame.loc[export_frame["source_row"].isin(exclusion_reasons.index), "source_row"].map(exclusion_reasons)
        )

    prediction_indexed = prediction_rows.set_index("source_row")
    prediction_mask = export_frame["source_row"].isin(prediction_indexed.index)
    export_frame.loc[prediction_mask, "predicted_label"] = (
        export_frame.loc[prediction_mask, "source_row"].map(prediction_indexed["predicted_label"])
    )
    export_frame.loc[prediction_mask, "probability_active"] = (
        export_frame.loc[prediction_mask, "source_row"].map(prediction_indexed["probability_active"])
    )
    export_frame.loc[prediction_mask, "probability_inactive"] = (
        export_frame.loc[prediction_mask, "source_row"].map(prediction_indexed["probability_inactive"])
    )

    export_frame = export_frame.drop(columns=["source_row"])

    csv_path = output_dir / "predictions_with_scores.csv"
    export_frame.to_csv(csv_path, sep=delimiter, index=False)

    outputs = {"csv": csv_path}
    if input_path.suffix.lower() == ".json":
        json_path = output_dir / "predictions_with_scores.json"
        json_path.write_text(export_frame.to_json(orient="records", indent=2), encoding="utf-8")
        outputs["json"] = json_path

    return outputs


def build_custom_tables(frame: pd.DataFrame) -> CustomDatasetTables:
    drug_table = frame[["SMILES"]].drop_duplicates().reset_index(drop=True)
    drug_table.insert(0, "drug_id", [f"drug_{index:06d}" for index in range(len(drug_table))])

    protein_table = frame[["SEQ"]].drop_duplicates().reset_index(drop=True)
    protein_table.insert(0, "protein_id", [f"protein_{index:06d}" for index in range(len(protein_table))])

    drug_lookup = drug_table.set_index("SMILES")["drug_id"]
    protein_lookup = protein_table.set_index("SEQ")["protein_id"]

    relation_table = frame.copy()
    relation_table.insert(0, "Drug_ID", frame["SMILES"].map(drug_lookup))
    relation_table.insert(1, "Prot_ID", frame["SEQ"].map(protein_lookup))
    relation_table = relation_table[["Drug_ID", "Prot_ID", "label"]]

    X_drug = drug_table.rename(columns={"drug_id": "drug_id", "SMILES": "SMILES"}).set_index("drug_id")
    X_target = protein_table.rename(columns={"protein_id": "protein_id", "SEQ": "SEQ"}).set_index("protein_id")

    return CustomDatasetTables(X_drug=X_drug, X_target=X_target, DTI=relation_table)


def save_custom_tables(tables: CustomDatasetTables, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    drug_path = output_dir / "drug_table.tsv"
    protein_path = output_dir / "protein_table.tsv"
    relation_path = output_dir / "relation_table.tsv"
    tables.X_drug.reset_index(names=["drug_id"]).to_csv(drug_path, sep="\t", index=False)
    tables.X_target.reset_index(names=["protein_id"]).to_csv(protein_path, sep="\t", index=False)
    tables.DTI.to_csv(relation_path, sep="\t", index=False)
    return {"drug_table": drug_path, "protein_table": protein_path, "relation_table": relation_path}


def scenario_training_overrides(
    args: argparse.Namespace,
    experiment: Any,
    run_tag: str,
    checkpoint_dir: Path,
) -> list[str]:
    overrides = [
        "tuning.param_search.tune=False",
        f"best_param_name={experiment.best_param_name}",
        f"callbacks.model_checkpoint.dirpath={normalize_dir(checkpoint_dir)}",
        "callbacks.model_checkpoint.save_top_k=1",
        "callbacks.model_checkpoint.save_last=True",
        "callbacks.model_checkpoint.save_weights_only=False",
        f"datamodule.serializer.save_path={normalize_dir(args.serialized_dir)}",
        f"logger.name={experiment.dataset}",
        f"hydra.run.dir={normalize_dir(args.artifacts_dir / 'hydra_runs' / experiment.dataset / experiment.scenario_key / run_tag)}",
        "hydra.output_subdir=null",
        "hydra.job.chdir=False",
    ]

    if experiment.dataset == "drugbank":
        overrides.append(f"preprocess.data_path={normalize_dir(args.drugbank_root)}")
    elif experiment.dataset == "bindingDB":
        overrides.append(f"preprocess.data_path={normalize_dir(args.bindingdb_root)}")
    elif experiment.dataset == "yamanishi":
        overrides.append(f"preprocess.root_path={normalize_dir(args.yamanishi_root)}")
        overrides.append("multiprocessing.multiprocessing=True")
        overrides.append("multiprocessing.num_process=10")
        overrides.append("multiprocessing.concurrent_process=1")
    elif experiment.dataset == "luo":
        overrides.append(f"preprocess.root_path={normalize_dir(args.luo_root)}")
        overrides.append("multiprocessing.multiprocessing=True")
        overrides.append("multiprocessing.num_process=10")
        overrides.append("multiprocessing.concurrent_process=1")

    overrides.extend(experiment.overrides)

    if experiment.dataset in {"drugbank", "bindingDB"}:
        overrides.append(f"datamodule.splitting.seed={args.seed}")

    return overrides


def run_command(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("COMMAND: " + shlex.join(command) + "\n\n")
        handle.flush()
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {shlex.join(command)}")


def parse_val_auc(path: Path) -> float:
    match = re.search(r"val_auc=([0-9]+\.[0-9]+)", path.name)
    if match:
        return float(match.group(1))
    return float("-inf")


def pick_checkpoint(checkpoint_dir: Path) -> Path:
    checkpoints = sorted(checkpoint_dir.glob("*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints were found in {checkpoint_dir}")

    non_last = [path for path in checkpoints if path.name != "last.ckpt"]
    if non_last:
        non_last.sort(key=lambda path: (parse_val_auc(path), path.stat().st_mtime), reverse=True)
        return non_last[0]

    checkpoints.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return checkpoints[0]


def compose_cfg(config_name: str, overrides: list[str]) -> Any:
    with hydra.initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base="1.3"):
        return hydra.compose(config_name=strip_yaml_suffix(config_name), overrides=overrides)


def update_best_params(cfg: Any) -> dict[str, Any]:
    from utils import utils

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        raise TypeError("Expected Hydra config to resolve to a dictionary.")
    return utils.update_best_param(cfg_dict)


def custom_serializer_names(custom_data_path: Path, scenario: str) -> tuple[str, str]:
    stem = sanitize_slug(custom_data_path.stem)
    scenario_slug = sanitize_slug(scenario.replace(":", "_"))
    return (
        f"custom_{stem}_{scenario_slug}_PubChem10M.pt",
        f"custom_{stem}_{scenario_slug}_ESM.pt",
    )


def resolve_training_serializer_paths(cfg: Any) -> tuple[Path, Path]:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        raise TypeError("Expected Hydra config to resolve to a dictionary.")
    serializer_cfg = cfg_dict["datamodule"]["serializer"]
    save_path = Path(serializer_cfg["save_path"])
    return save_path / serializer_cfg["drug_name"], save_path / serializer_cfg["target_name"]


def relax_transformers_torch_load_guard() -> None:
    # The ChemBERTa checkpoint used by this repo is distributed as a legacy
    # PyTorch bin file. Recent transformers releases block loading those files
    # on torch<2.6, even for trusted local workflows such as this Colab setup.
    try:
        import transformers.modeling_utils as modeling_utils
        from transformers.utils import import_utils as transformers_import_utils
    except ImportError:
        return

    if hasattr(transformers_import_utils, "check_torch_load_is_safe"):
        transformers_import_utils.check_torch_load_is_safe = lambda: None
    if hasattr(modeling_utils, "check_torch_load_is_safe"):
        modeling_utils.check_torch_load_is_safe = lambda: None


def ensure_training_embeddings(
    experiment: Any,
    config_name: str,
    training_overrides: list[str],
    artifacts_root: Path,
) -> None:
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
    command = [sys.executable, str(RUN_PY), "--config-name", experiment.config_name, *bootstrap_overrides]
    run_command(command, bootstrap_log)

    if not drug_path.exists() or not target_path.exists():
        raise FileNotFoundError(
            "Built-in scenario embeddings were not created as expected: "
            f"{drug_path}, {target_path}"
        )


def generate_custom_embeddings(
    cfg: Any,
    tables: CustomDatasetTables,
    serialized_dir: Path,
    drug_name: str,
    target_name: str,
    reuse_existing: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Path]]:
    serialized_dir.mkdir(parents=True, exist_ok=True)
    drug_path = serialized_dir / drug_name
    target_path = serialized_dir / target_name

    if reuse_existing and drug_path.exists() and target_path.exists():
        X_drug = torch.load(drug_path, map_location="cpu")
        X_target = torch.load(target_path, map_location="cpu")
        return X_drug, X_target, {"drug_embedding": drug_path, "protein_embedding": target_path}

    relax_transformers_torch_load_guard()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    drug_featurizer = hydra.utils.instantiate(cfg["featurizer"]["drugfeaturizer"], device, _recursive_=False)
    prot_featurizer = hydra.utils.instantiate(cfg["featurizer"]["protfeaturizer"], device, _recursive_=False)

    drug_features = drug_featurizer.get_representations(tables.X_drug.SMILES.values)
    target_features = prot_featurizer.get_representations(tables.X_target.SEQ.values)
    X_drug = pd.DataFrame(drug_features, index=tables.X_drug.index)
    X_target = pd.DataFrame(target_features, index=tables.X_target.index)
    torch.save(X_drug, drug_path)
    torch.save(X_target, target_path)
    return X_drug, X_target, {"drug_embedding": drug_path, "protein_embedding": target_path}


def build_custom_eval_dataset(
    tables: CustomDatasetTables,
    X_drug_embeddings: pd.DataFrame,
    X_target_embeddings: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    drug_index = pd.Series(range(len(X_drug_embeddings)), index=X_drug_embeddings.index)
    target_index = pd.Series(range(len(X_target_embeddings)), index=X_target_embeddings.index)

    test_table = tables.DTI.copy()
    test_table["Drug_ID"] = test_table["Drug_ID"].map(drug_index)
    test_table["Prot_ID"] = test_table["Prot_ID"].map(target_index)
    if test_table[["Drug_ID", "Prot_ID"]].isna().any().any():
        raise ValueError("Could not map one or more custom drug/protein identifiers to embedding indices.")
    test_table["Drug_ID"] = test_table["Drug_ID"].astype(int)
    test_table["Prot_ID"] = test_table["Prot_ID"].astype(int)

    empty = test_table.iloc[0:0].copy()
    return {
        "X_drug": X_drug_embeddings,
        "X_target": X_target_embeddings,
        "train": empty,
        "val": empty,
        "test": test_table,
        "ddi": None,
    }


def build_prediction_dataset(
    tables: CustomDatasetTables,
    X_drug_embeddings: pd.DataFrame,
    X_target_embeddings: pd.DataFrame,
    relation_table: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    drug_index = pd.Series(range(len(X_drug_embeddings)), index=X_drug_embeddings.index)
    target_index = pd.Series(range(len(X_target_embeddings)), index=X_target_embeddings.index)

    test_table = relation_table.copy()
    test_table["Drug_ID"] = test_table["Drug_ID"].map(drug_index)
    test_table["Prot_ID"] = test_table["Prot_ID"].map(target_index)
    if test_table[["Drug_ID", "Prot_ID"]].isna().any().any():
        raise ValueError("Could not map one or more custom drug/protein identifiers to embedding indices.")
    test_table["Drug_ID"] = test_table["Drug_ID"].astype(int)
    test_table["Prot_ID"] = test_table["Prot_ID"].astype(int)

    empty = test_table.iloc[0:0].copy()
    return {
        "X_drug": X_drug_embeddings,
        "X_target": X_target_embeddings,
        "train": empty,
        "val": empty,
        "test": test_table,
        "ddi": None,
    }


def export_checkpoint_to_safetensors(checkpoint_path: Path, output_path: Path, metadata: dict[str, str]) -> None:
    try:
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover - runtime environment guard
        raise RuntimeError("safetensors is required to export model weights. Install it in the active environment.") from exc

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    cpu_state_dict = {key: value.detach().cpu().contiguous() for key, value in state_dict.items()}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(cpu_state_dict, str(output_path), metadata=metadata)


def predict_checkpoint_on_custom_dataset(
    cfg_dict: dict[str, Any],
    checkpoint_path: Path,
    dataset: dict[str, pd.DataFrame],
    source_rows: list[int],
) -> pd.DataFrame:
    from datamodule.dataloader_GAT import MyDataset
    from module.cognn_cross import Net

    model = Net.load_from_checkpoint(
        str(checkpoint_path),
        cfg=cfg_dict,
        dataset=dataset,
        network=cfg_dict["module"]["network"],
        optimizer=cfg_dict["module"]["optimizer"],
        criterion=cfg_dict["module"]["criterion"],
        GAT_params=cfg_dict["module"]["GAT_params"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    test_table = dataset["test"].reset_index(drop=True)
    prediction_dataset = MyDataset(dataset["X_drug"], dataset["X_target"], test_table)
    dataloader = torch.utils.data.DataLoader(
        prediction_dataset,
        batch_size=cfg_dict["datamodule"]["dm_cfg"]["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    probabilities: list[float] = []
    true_labels: list[int] = []

    with torch.no_grad():
        for batch in dataloader:
            x1, x2, y, drugs, targets = batch
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device)
            drugs = drugs.to(device)
            targets = targets.to(device)
            x1_org, x2_org = x1, x2
            x1_proc, x2_proc, x1_network, x2_network, inv_drug, inv_target = model.common_preprocess(
                x1, x2, drugs, targets, 0
            )
            x1_proc = x1_proc.to(device)
            x2_proc = x2_proc.to(device)
            x1_network = x1_network.to(device)
            x2_network = x2_network.to(device)
            inv_drug = inv_drug.to(device)
            inv_target = inv_target.to(device)

            logits, _, _ = model.forward(
                x1_proc,
                x2_proc,
                x1_org,
                x2_org,
                x1_network,
                x2_network,
                inv_drug,
                inv_target,
                y,
            )
            batch_probabilities = torch.sigmoid(logits).detach().cpu().numpy().tolist()
            probabilities.extend(batch_probabilities)
            true_labels.extend(y.detach().cpu().numpy().astype(int).tolist())

    predicted_labels = [1 if probability >= 0.5 else 0 for probability in probabilities]
    probability_inactive = [1.0 - probability for probability in probabilities]

    return pd.DataFrame(
        {
            "source_row": source_rows,
            "true_label": true_labels,
            "predicted_label": predicted_labels,
            "probability_active": probabilities,
            "probability_inactive": probability_inactive,
        }
    )


def ensure_repo_on_path() -> None:
    repo_root_str = str(REPO_ROOT)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


def main() -> None:
    args = parse_args()
    ensure_repo_on_path()

    reproduce_module = load_reproduce_paper_module()
    experiment = get_experiment(reproduce_module, args.scenario)

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
    training_overrides = scenario_training_overrides(args, experiment, run_tag, checkpoint_dir)
    ensure_training_embeddings(experiment, experiment.config_name, training_overrides, run_root)

    if not args.skip_training:
        train_log_path = logs_dir / "train.log"
        command = [sys.executable, str(RUN_PY), "--config-name", experiment.config_name, *training_overrides]
        run_command(command, train_log_path)

    checkpoint_path = pick_checkpoint(checkpoint_dir)

    eval_cfg = compose_cfg(experiment.config_name, training_overrides)
    eval_cfg_dict = update_best_params(eval_cfg)

    custom_drug_name, custom_target_name = custom_serializer_names(args.custom_data, args.scenario)
    X_drug_embeddings, X_target_embeddings, embedding_paths = generate_custom_embeddings(
        eval_cfg,
        tables,
        args.serialized_dir,
        custom_drug_name,
        custom_target_name,
        args.reuse_custom_embeddings,
    )

    relations_with_source = filtered_custom_data.frame[["source_row"]].copy()
    relations_with_source["Drug_ID"] = tables.DTI["Drug_ID"].values
    relations_with_source["Prot_ID"] = tables.DTI["Prot_ID"].values
    relations_with_source["label"] = tables.DTI["label"].values
    prediction_dataset = build_prediction_dataset(
        tables,
        X_drug_embeddings,
        X_target_embeddings,
        relations_with_source[["Drug_ID", "Prot_ID", "label"]],
    )
    prediction_rows = predict_checkpoint_on_custom_dataset(
        eval_cfg_dict,
        checkpoint_path,
        prediction_dataset,
        relations_with_source["source_row"].tolist(),
    )
    prediction_exports = save_prediction_export(
        filtered_custom_data,
        prediction_rows,
        export_dir,
        args.custom_data,
        args.delimiter,
    )

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
