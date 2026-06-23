from pathlib import Path

import pandas as pd

from scripts.custom_dataset_utils import (
    FilteredCustomData,
    build_custom_tables,
    load_cached_inference_preparation,
    save_custom_tables,
    save_inference_preparation_manifest,
)


def test_cache_round_trip_preserves_literal_na_protein_sequence(tmp_path: Path) -> None:
    input_path = tmp_path / "pairs.parquet"
    input_path.write_bytes(b"source identity for cache manifest")
    prepared_dir = tmp_path / "prepared_data"

    frame = pd.DataFrame({"source_row": [0], "SMILES": ["CC"], "SEQ": ["NA"], "label": [0]})
    filtered = FilteredCustomData(
        frame=frame,
        excluded=pd.DataFrame(columns=["source_row", "reason", "SMILES", "SEQ", "label"]),
        original=frame.copy(),
    )
    tables = build_custom_tables(filtered.frame)
    prepared = {
        "filtered": filtered,
        "tables": tables,
        "prepared_paths": save_custom_tables(tables, prepared_dir),
        "exclusions_report": None,
        "relations_with_source": tables.DTI.assign(source_row=filtered.frame["source_row"].values),
    }
    save_inference_preparation_manifest(prepared, input_path, ",", True, prepared_dir)

    cached = load_cached_inference_preparation(input_path, ",", True, prepared_dir)

    assert cached is not None
    assert cached["filtered"].frame["SEQ"].tolist() == ["NA"]
