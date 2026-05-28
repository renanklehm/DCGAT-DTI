from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.common import relax_transformers_torch_load_guard, sanitize_slug


@dataclass(frozen=True)
class CustomDatasetTables:
    X_drug: Any
    X_target: Any
    DTI: Any


@dataclass(frozen=True)
class FilteredCustomData:
    frame: Any
    excluded: Any
    original: Any


def read_custom_triplets(path: Path, delimiter: str, has_header: bool) -> FilteredCustomData:
    import pandas as pd

    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path, sep=delimiter, header=0 if has_header else None)
    elif suffix == ".json":
        frame = pd.read_json(path)
    elif suffix == ".parquet":
        frame = pd.read_parquet(path)
    else:
        raise ValueError("Custom dataset must be a .csv, .json, or .parquet file.")

    if frame.empty:
        raise ValueError(f"No data rows were found in {path}")

    if suffix in {".json", ".parquet"}:
        expected_columns = {"smiles", "sequence", "activation"}
        lowered = {str(column).strip().lower(): column for column in frame.columns}
        if not expected_columns.issubset(lowered):
            raise ValueError(f"{suffix} input must contain columns named smiles, sequence, and activation.")
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


def save_exclusions_report(excluded: Any, output_dir: Path) -> dict[str, Path] | None:
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


def build_custom_tables(frame: Any) -> CustomDatasetTables:
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

    x_drug = drug_table.rename(columns={"SMILES": "SMILES"}).set_index("drug_id")
    x_target = protein_table.rename(columns={"SEQ": "SEQ"}).set_index("protein_id")

    return CustomDatasetTables(X_drug=x_drug, X_target=x_target, DTI=relation_table)


def save_custom_tables(tables: CustomDatasetTables, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    drug_path = output_dir / "drug_table.tsv"
    protein_path = output_dir / "protein_table.tsv"
    relation_path = output_dir / "relation_table.tsv"
    tables.X_drug.reset_index(names=["drug_id"]).to_csv(drug_path, sep="\t", index=False)
    tables.X_target.reset_index(names=["protein_id"]).to_csv(protein_path, sep="\t", index=False)
    tables.DTI.to_csv(relation_path, sep="\t", index=False)
    return {"drug_table": drug_path, "protein_table": protein_path, "relation_table": relation_path}


def custom_serializer_names(custom_data_path: Path, suffix: str) -> tuple[str, str]:
    stem = sanitize_slug(custom_data_path.stem)
    suffix_slug = sanitize_slug(suffix)
    return (
        f"custom_{stem}_{suffix_slug}_PubChem10M.pt",
        f"custom_{stem}_{suffix_slug}_ESM.pt",
    )


def generate_custom_embeddings(
    cfg: Any,
    tables: CustomDatasetTables,
    serialized_dir: Path,
    drug_name: str,
    target_name: str,
    reuse_existing: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Path]]:
    import hydra
    import pandas as pd
    import torch

    serialized_dir.mkdir(parents=True, exist_ok=True)
    drug_path = serialized_dir / drug_name
    target_path = serialized_dir / target_name

    if reuse_existing and drug_path.exists() and target_path.exists():
        x_drug = torch.load(drug_path, map_location="cpu")
        x_target = torch.load(target_path, map_location="cpu")
        return x_drug, x_target, {"drug_embedding": drug_path, "protein_embedding": target_path}

    relax_transformers_torch_load_guard()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    drug_featurizer = hydra.utils.instantiate(cfg["featurizer"]["drugfeaturizer"], device, _recursive_=False)
    prot_featurizer = hydra.utils.instantiate(cfg["featurizer"]["protfeaturizer"], device, _recursive_=False)

    drug_features = drug_featurizer.get_representations(tables.X_drug.SMILES.values)
    target_features = prot_featurizer.get_representations(tables.X_target.SEQ.values)
    x_drug = pd.DataFrame(drug_features, index=tables.X_drug.index)
    x_target = pd.DataFrame(target_features, index=tables.X_target.index)
    torch.save(x_drug, drug_path)
    torch.save(x_target, target_path)
    return x_drug, x_target, {"drug_embedding": drug_path, "protein_embedding": target_path}


def build_prediction_dataset(
    x_drug_embeddings: Any,
    x_target_embeddings: Any,
    relation_table: Any,
) -> dict[str, Any]:
    import pandas as pd

    drug_index = pd.Series(range(len(x_drug_embeddings)), index=x_drug_embeddings.index)
    target_index = pd.Series(range(len(x_target_embeddings)), index=x_target_embeddings.index)

    test_table = relation_table.copy()
    test_table["Drug_ID"] = test_table["Drug_ID"].map(drug_index)
    test_table["Prot_ID"] = test_table["Prot_ID"].map(target_index)
    if test_table[["Drug_ID", "Prot_ID"]].isna().any().any():
        raise ValueError("Could not map one or more custom drug/protein identifiers to embedding indices.")
    test_table["Drug_ID"] = test_table["Drug_ID"].astype(int)
    test_table["Prot_ID"] = test_table["Prot_ID"].astype(int)

    empty = test_table.iloc[0:0].copy()
    return {
        "X_drug": x_drug_embeddings,
        "X_target": x_target_embeddings,
        "train": empty,
        "val": empty,
        "test": test_table,
        "ddi": None,
    }


def save_prediction_export(
    filtered_custom_data: FilteredCustomData,
    prediction_rows: Any,
    output_dir: Path,
    input_path: Path,
    delimiter: str,
    split_assignments: Any = None,
) -> dict[str, Path]:
    import numpy as np
    import pandas as pd

    output_dir.mkdir(parents=True, exist_ok=True)

    export_frame = filtered_custom_data.original.copy()
    export_frame = export_frame.rename(columns={"SMILES": "smiles", "SEQ": "sequence", "label": "activity"})
    export_frame["excluded_reason"] = ""
    export_frame["predicted_label"] = pd.Series([pd.NA] * len(export_frame), dtype="Int64")
    export_frame["probability_active"] = np.nan
    export_frame["probability_inactive"] = np.nan
    if split_assignments is not None:
        export_frame["split"] = "excluded"

    if not filtered_custom_data.excluded.empty:
        exclusion_reasons = filtered_custom_data.excluded.set_index("source_row")["reason"]
        excluded_mask = export_frame["source_row"].isin(exclusion_reasons.index)
        export_frame.loc[excluded_mask, "excluded_reason"] = export_frame.loc[excluded_mask, "source_row"].map(exclusion_reasons)

    prediction_indexed = prediction_rows.set_index("source_row")
    prediction_mask = export_frame["source_row"].isin(prediction_indexed.index)
    export_frame.loc[prediction_mask, "predicted_label"] = export_frame.loc[prediction_mask, "source_row"].map(
        prediction_indexed["predicted_label"]
    )
    export_frame.loc[prediction_mask, "probability_active"] = export_frame.loc[prediction_mask, "source_row"].map(
        prediction_indexed["probability_active"]
    )
    export_frame.loc[prediction_mask, "probability_inactive"] = export_frame.loc[prediction_mask, "source_row"].map(
        prediction_indexed["probability_inactive"]
    )
    if split_assignments is not None:
        export_frame.loc[prediction_mask, "split"] = (
            export_frame.loc[prediction_mask, "source_row"].map(split_assignments).fillna("unused")
        )

    export_frame = export_frame.drop(columns=["source_row"])

    csv_path = output_dir / "predictions_with_scores.csv"
    export_frame.to_csv(csv_path, sep=delimiter, index=False)

    outputs = {"csv": csv_path}
    if input_path.suffix.lower() in {".json", ".parquet"}:
        json_path = output_dir / "predictions_with_scores.json"
        json_path.write_text(export_frame.to_json(orient="records", indent=2), encoding="utf-8")
        outputs["json"] = json_path

    return outputs


def export_checkpoint_to_safetensors(checkpoint_path: Path, output_path: Path, metadata: dict[str, str]) -> None:
    import torch

    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise RuntimeError("safetensors is required to export model weights. Install it in the active environment.") from exc

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    cpu_state_dict = {key: value.detach().cpu().contiguous() for key, value in state_dict.items()}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(cpu_state_dict, str(output_path), metadata=metadata)


def predict_checkpoint_on_dataset(
    cfg_dict: dict[str, Any],
    checkpoint_path: Path,
    dataset: dict[str, pd.DataFrame],
    source_rows: list[int],
) -> pd.DataFrame:
    import pandas as pd
    import torch

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
