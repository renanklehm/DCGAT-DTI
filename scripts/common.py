from __future__ import annotations

import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
        scenario_key="balanced_cold_full",
        paper_label="Balanced Cold Start for Drug and Protein",
        repeats=10,
        # There is no cold_full-specific tuned config yet, so reuse the closest
        # existing cold-start GAT parameters.
        best_param_name="cold_drug_balanced_GAT.yaml",
        overrides=(
            "datamodule.splitting.balanced=True",
            "datamodule.splitting.splitting_strategy=cold_full",
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
        dataset="drugbank",
        config_name="drugbank_train_GAT.yaml",
        scenario_key="unbalanced_cold_full",
        paper_label="Unbalanced Cold Start for Drug and Protein",
        repeats=10,
        best_param_name="cold_drug_unbalanced.yaml",
        overrides=(
            "datamodule.splitting.balanced=False",
            "datamodule.splitting.splitting_strategy=cold_full",
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
        scenario_key="balanced_cold_full",
        paper_label="Balanced Cold Start for Drug and Protein",
        repeats=10,
        best_param_name="cold_drug_balanced_GAT.yaml",
        overrides=(
            "datamodule.splitting.balanced=True",
            "datamodule.splitting.splitting_strategy=cold_full",
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
    Experiment(
        dataset="bindingDB",
        config_name="bindingDB_train_GAT.yaml",
        scenario_key="unbalanced_cold_full",
        paper_label="Unbalanced Cold Start for Drug and Protein",
        repeats=10,
        best_param_name="cold_drug_unbalanced.yaml",
        overrides=(
            "datamodule.splitting.balanced=False",
            "datamodule.splitting.splitting_strategy=cold_full",
            "multiprocessing.multiprocessing=False",
        ),
    ),
)


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

SPLIT_STRATEGY_ALIASES = {
    "warm": "random",
    "random": "random",
    "cold_drug": "cold_drug",
    "cold_target": "cold_target",
    "cold_full": "cold_full",
}

SERIALIZED_FILES = {
    "drugbank": ("DrugBank_PubChem10M.pt", "DrugBank_ESM.pt"),
    "bindingDB": ("bindingDB_Kd_PubChem10M.pt", "bindingDB_Kd_ESM.pt"),
    "yamanishi": ("yamanishi_PubChem10M.pt", "yamanishi_ESM.pt"),
    "luo": ("luo_PubChem10M.pt", "luo_ESM.pt"),
}


def ensure_repo_on_path() -> None:
    repo_root_str = str(REPO_ROOT)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


def normalize_dir(path: Path) -> str:
    return path.resolve().as_posix().rstrip("/") + "/"


def strip_yaml_suffix(config_name: str) -> str:
    return config_name[:-5] if config_name.endswith(".yaml") else config_name


def sanitize_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return slug.strip("._-") or "custom"


def normalize_split_strategy(value: str) -> str:
    try:
        return SPLIT_STRATEGY_ALIASES[value]
    except KeyError as exc:
        available = ", ".join(sorted(SPLIT_STRATEGY_ALIASES))
        raise ValueError(f"Unknown split strategy '{value}'. Available values: {available}") from exc


def default_best_param_name(split_strategy: str, balanced: bool) -> str:
    normalize_split_strategy(split_strategy)
    return "custom_default.yaml"


def get_experiment(scenario: str) -> Experiment:
    if ":" not in scenario:
        raise ValueError("Scenario must be provided as dataset:scenario_key.")
    dataset, scenario_key = scenario.split(":", 1)
    for experiment in ALL_EXPERIMENTS:
        if experiment.dataset == dataset and experiment.scenario_key == scenario_key:
            return experiment
    available = sorted(f"{exp.dataset}:{exp.scenario_key}" for exp in ALL_EXPERIMENTS)
    raise ValueError(f"Unknown scenario '{scenario}'. Available scenarios: {', '.join(available)}")


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
    import hydra

    with hydra.initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base="1.3"):
        return hydra.compose(config_name=strip_yaml_suffix(config_name), overrides=overrides)


def update_best_params(cfg: Any) -> dict[str, Any]:
    from omegaconf import OmegaConf
    from utils import utils

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        raise TypeError("Expected Hydra config to resolve to a dictionary.")
    return utils.update_best_param(cfg_dict)


def load_checkpoint_cfg_dict(checkpoint_path: Path) -> dict[str, Any] | None:
    import torch
    from omegaconf import OmegaConf

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    hyper_parameters = checkpoint.get("hyper_parameters")
    if hyper_parameters is None:
        return None

    candidate = hyper_parameters.get("cfg", hyper_parameters) if isinstance(hyper_parameters, dict) else hyper_parameters
    if not isinstance(candidate, dict):
        candidate = OmegaConf.to_container(candidate, resolve=True)
    if not isinstance(candidate, dict):
        return None
    if "module" not in candidate or "datamodule" not in candidate:
        return None
    return candidate


def resolve_training_serializer_paths(cfg: Any) -> tuple[Path, Path]:
    from omegaconf import OmegaConf

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        raise TypeError("Expected Hydra config to resolve to a dictionary.")
    serializer_cfg = cfg_dict["datamodule"]["serializer"]
    save_path = Path(serializer_cfg["save_path"])
    return save_path / serializer_cfg["drug_name"], save_path / serializer_cfg["target_name"]


def relax_transformers_torch_load_guard() -> None:
    try:
        import transformers.modeling_utils as modeling_utils
        from transformers.utils import import_utils as transformers_import_utils
    except ImportError:
        return

    if hasattr(transformers_import_utils, "check_torch_load_is_safe"):
        transformers_import_utils.check_torch_load_is_safe = lambda: None
    if hasattr(modeling_utils, "check_torch_load_is_safe"):
        modeling_utils.check_torch_load_is_safe = lambda: None
