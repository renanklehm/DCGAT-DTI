from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from scripts.custom_dataset_utils import read_custom_triplets
from scripts.train_custom_dataset_and_export import (
    build_gsdti_aligned_dataset,
    load_gsdti_holdout_source_rows,
)
from main import resolve_delimiter


def test_gsdti_training_defaults_to_comma_delimiter() -> None:
    args = SimpleNamespace(split_strategy="gsdti", train_delimiter=None, delimiter="|")

    assert resolve_delimiter(args, "train") == ","


def test_read_custom_triplets_accepts_gsdti_canonical_table(tmp_path: Path) -> None:
    path = tmp_path / "df_less1000.csv"
    pd.DataFrame(
        {
            "Drug_ID": ["NPASS_D1", "NPASS_D2"],
            "Drug": ["CC", "CCC"],
            "Target_ID": ["NPASS_T1", "NPASS_T2"],
            "Target": ["ACDE", "FGHI"],
            "Y": [0.0, 1.0],
            "Label": [0, 1],
            "Target_Length": [4, 4],
        }
    ).to_csv(path, index=False)

    result = read_custom_triplets(path, ",", True)

    assert result.frame[["SMILES", "SEQ", "label"]].to_dict("records") == [
        {"SMILES": "CC", "SEQ": "ACDE", "label": 0},
        {"SMILES": "CCC", "SEQ": "FGHI", "label": 1},
    ]


def test_gsdti_aligned_split_recreates_canonical_holdout_before_dcgat_filtering() -> None:
    row_count = 100
    original = pd.DataFrame(
        {
            "source_row": np.arange(row_count),
            "SMILES": [f"C{index}" for index in range(row_count)],
            "SEQ": ["ACDE"] * row_count,
            "label": np.tile([0, 1], row_count // 2),
        }
    )
    excluded_source_row = 7
    kept = original[original["source_row"] != excluded_source_row]
    x_drug = pd.DataFrame(index=[f"drug_{index}" for index in range(row_count)])
    x_target = pd.DataFrame(index=["protein_0"])
    relations = pd.DataFrame(
        {
            "Drug_ID": [f"drug_{index}" for index in kept["source_row"]],
            "Prot_ID": "protein_0",
            "label": kept["label"].to_numpy(),
            "source_row": kept["source_row"].to_numpy(),
        }
    )

    dataset = build_gsdti_aligned_dataset(
        x_drug,
        x_target,
        relations,
        original,
        gsdti_validation_size=0.10,
        dcgat_validation_size=0.10,
        seed=42,
        balanced=False,
        unbalanced_ratio=0,
    )

    _, expected_holdout = train_test_split(
        original["source_row"].to_numpy(),
        test_size=0.10,
        random_state=42,
        stratify=original["label"].to_numpy(),
    )
    expected_kept_holdout = set(expected_holdout) - {excluded_source_row}
    actual_holdout = set(dataset["test"]["source_row"].astype(int))
    assert actual_holdout == expected_kept_holdout

    memberships = [set(dataset[name]["source_row"].astype(int)) for name in ("train", "val", "test")]
    assert memberships[0].isdisjoint(memberships[1])
    assert memberships[0].isdisjoint(memberships[2])
    assert memberships[1].isdisjoint(memberships[2])
    assert set.union(*memberships) == set(kept["source_row"].astype(int))


def test_gsdti_prediction_export_can_define_realized_holdout(tmp_path: Path) -> None:
    original = pd.DataFrame(
        {
            "source_row": [0, 1, 2, 3],
            "SMILES": ["CC", "CCC", "CCCC", "CCCCC"],
            "SEQ": ["ACDE", "FGHI", "KLMN", "PQRS"],
            "label": [0, 1, 0, 1],
        }
    )
    predictions_path = tmp_path / "NPASS_predictions.csv"
    pd.DataFrame(
        {
            "smiles": ["CCC", "CCCCC"],
            "sequence": ["FGHI", "PQRS"],
            "label": [0, 1],
            "probability": [0.4, 0.8],
            "real_value": [1.0, 1.0],
        }
    ).to_csv(predictions_path, index=False)

    source_rows = load_gsdti_holdout_source_rows(original, predictions_path)

    assert source_rows.tolist() == [1, 3]
