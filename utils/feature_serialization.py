from __future__ import annotations

from pathlib import Path

import hydra
import pandas as pd
import torch


def materialize_serialized_features(
    cfg,
    X_drug: pd.DataFrame,
    X_target: pd.DataFrame,
    *,
    save_path: str | Path | None = None,
    drug_name: str | None = None,
    target_name: str | None = None,
    load_serialized: bool | None = None,
    device=None,
):
    serializer_cfg = cfg["datamodule"]["serializer"]
    save_dir = Path(save_path if save_path is not None else serializer_cfg["save_path"])
    save_dir.mkdir(parents=True, exist_ok=True)

    drug_file = save_dir / (drug_name if drug_name is not None else serializer_cfg["drug_name"])
    target_file = save_dir / (target_name if target_name is not None else serializer_cfg["target_name"])
    should_load = serializer_cfg["load_serialized"] if load_serialized is None else load_serialized

    if should_load:
        return torch.load(drug_file), torch.load(target_file)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    drug_featurizer = hydra.utils.instantiate(cfg["featurizer"]["drugfeaturizer"], device, _recursive_=False)
    X_drug_features = drug_featurizer.get_representations(X_drug.SMILES.values)
    X_drug = pd.DataFrame(X_drug_features, index=X_drug.index)
    torch.save(X_drug, drug_file)

    prot_featurizer = hydra.utils.instantiate(cfg["featurizer"]["protfeaturizer"], device, _recursive_=False)
    X_target_features = prot_featurizer.get_representations(X_target.SEQ.values)
    X_target = pd.DataFrame(X_target_features, index=X_target.index)
    torch.save(X_target, target_file)

    return X_drug, X_target
