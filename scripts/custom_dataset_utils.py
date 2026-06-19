from __future__ import annotations

import hashlib
import json
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.common import relax_transformers_torch_load_guard, sanitize_slug


PROGRESS_CHUNK_ROWS = 250_000
INFERENCE_PREPARATION_CACHE_VERSION = 1


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


def _inference_source_signature(path: Path, delimiter: str, has_header: bool) -> dict[str, Any]:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "sha256": digest.hexdigest(),
        "delimiter": delimiter,
        "has_header": has_header,
    }


def _inference_source_matches(cached: dict[str, Any], current: dict[str, Any]) -> bool:
    """Accept older manifests without treating a harmless mtime change as new input."""
    stable_keys = ("size", "delimiter", "has_header")
    if any(cached.get(key) != current.get(key) for key in stable_keys):
        return False
    cached_digest = cached.get("sha256")
    if cached_digest is not None:
        return cached_digest == current["sha256"]
    return cached.get("path") == current["path"]


def save_inference_preparation_manifest(
    prepared: dict[str, Any],
    input_path: Path,
    delimiter: str,
    has_header: bool,
    prepared_dir: Path,
) -> Path:
    manifest_path = prepared_dir / "inference_preparation_manifest.json"
    tracked_paths = dict(prepared["prepared_paths"])
    if prepared["exclusions_report"] is not None:
        tracked_paths.update({f"exclusions_{key}": value for key, value in prepared["exclusions_report"].items()})
    manifest = {
        "version": INFERENCE_PREPARATION_CACHE_VERSION,
        "source": _inference_source_signature(input_path, delimiter, has_header),
        "rows": {
            "kept": int(len(prepared["filtered"].frame)),
            "excluded": int(len(prepared["filtered"].excluded)),
        },
        "files": {key: {"name": path.name, "size": path.stat().st_size} for key, path in tracked_paths.items()},
    }
    prepared_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = manifest_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary_path.replace(manifest_path)
    return manifest_path


def load_cached_inference_preparation(
    input_path: Path,
    delimiter: str,
    has_header: bool,
    prepared_dir: Path,
    log_fn: Any | None = None,
) -> dict[str, Any] | None:
    """Load completed inference TSVs, including caches made before manifests existed."""
    import numpy as np
    import pandas as pd

    drug_path = prepared_dir / "drug_table.tsv"
    protein_path = prepared_dir / "protein_table.tsv"
    relation_path = prepared_dir / "relation_table.tsv"
    excluded_path = prepared_dir / "excluded_rows.tsv"
    summary_path = prepared_dir / "excluded_summary.json"
    manifest_path = prepared_dir / "inference_preparation_manifest.json"
    required_table_paths = (drug_path, protein_path, relation_path)

    if not all(path.is_file() for path in required_table_paths):
        return None

    manifest = None
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("version") != INFERENCE_PREPARATION_CACHE_VERSION:
                _emit(log_fn, "Prepared inference cache version changed; rebuilding")
                return None
            current_source = _inference_source_signature(input_path, delimiter, has_header)
            if not _inference_source_matches(manifest.get("source", {}), current_source):
                _emit(log_fn, "Prepared inference cache does not match the current input; rebuilding")
                return None
            for file_info in manifest.get("files", {}).values():
                cached_path = prepared_dir / file_info["name"]
                if not cached_path.is_file() or cached_path.stat().st_size != file_info["size"]:
                    _emit(log_fn, f"Prepared inference cache file is missing or incomplete: {cached_path}; rebuilding")
                    return None
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            _emit(log_fn, f"Prepared inference cache manifest is invalid ({exc}); rebuilding")
            return None
    elif input_path.suffix.lower() != ".parquet" or not (excluded_path.is_file() and summary_path.is_file()):
        # A pre-manifest cache can only prove completeness when its exclusions outputs exist too.
        return None

    cache_kind = "validated" if manifest is not None else "legacy"
    _emit(log_fn, f"Reusing {cache_kind} prepared inference TSVs from {prepared_dir}")
    try:
        with _timed_step(log_fn, "Loading prepared inference TSVs"):
            drug_table = pd.read_csv(drug_path, sep="\t", dtype=str)
            protein_table = pd.read_csv(protein_path, sep="\t", dtype=str)
            relation_table = pd.read_csv(
                relation_path,
                sep="\t",
                dtype={"Drug_ID": str, "Prot_ID": str, "label": int},
            )
            if excluded_path.is_file():
                excluded = pd.read_csv(excluded_path, sep="\t")
            else:
                excluded = pd.DataFrame(columns=["source_row", "reason", "SMILES", "SEQ", "label"])

        expected_columns = (
            {"drug_id", "SMILES"},
            {"protein_id", "SEQ"},
            {"Drug_ID", "Prot_ID", "label"},
        )
        actual_columns = (set(drug_table.columns), set(protein_table.columns), set(relation_table.columns))
        if actual_columns != expected_columns:
            raise ValueError("prepared table columns do not match the inference cache schema")

        kept_count = len(relation_table)
        excluded_count = len(excluded)
        if manifest is not None and manifest.get("rows") != {"kept": kept_count, "excluded": excluded_count}:
            raise ValueError("prepared table row counts do not match the manifest")
        if manifest is None:
            import pyarrow.parquet as pq

            source_row_count = pq.ParquetFile(input_path).metadata.num_rows
            if source_row_count != kept_count + excluded_count:
                raise ValueError("legacy prepared table row counts do not match the Parquet input")

        excluded["source_row"] = excluded["source_row"].astype(int)
        total_count = kept_count + excluded_count
        excluded_source_rows = excluded["source_row"].to_numpy(dtype=np.int64, copy=False)
        if excluded_count and ((excluded_source_rows < 0).any() or (excluded_source_rows >= total_count).any()):
            raise ValueError("excluded source rows are not a valid subset of the original input")
        excluded_mask = np.zeros(total_count, dtype=bool)
        excluded_mask[excluded_source_rows] = True
        if int(excluded_mask.sum()) != excluded_count:
            raise ValueError("excluded source rows contain duplicates")
        kept_source_rows = np.flatnonzero(~excluded_mask)
        if kept_source_rows.size != kept_count:
            raise ValueError("could not reconstruct kept source rows")

        drug_lookup = drug_table.set_index("drug_id")["SMILES"]
        protein_lookup = protein_table.set_index("protein_id")["SEQ"]
        kept = pd.DataFrame(
            {
                "source_row": kept_source_rows,
                "SMILES": relation_table["Drug_ID"].map(drug_lookup),
                "SEQ": relation_table["Prot_ID"].map(protein_lookup),
                "label": relation_table["label"].astype(int),
            }
        )
        if kept[["SMILES", "SEQ"]].isna().any().any():
            raise ValueError("relation table references unknown drug or protein IDs")

        excluded["label"] = excluded["label"].astype(int)
        original = pd.concat(
            [kept, excluded[["source_row", "SMILES", "SEQ", "label"]]],
            ignore_index=True,
        ).sort_values("source_row", ignore_index=True)
        tables = CustomDatasetTables(
            X_drug=drug_table.set_index("drug_id"),
            X_target=protein_table.set_index("protein_id"),
            DTI=relation_table,
        )
        relations_with_source = relation_table.copy()
        relations_with_source["source_row"] = kept_source_rows
        prepared_paths = {"drug_table": drug_path, "protein_table": protein_path, "relation_table": relation_path}
        exclusions_report = None
        if excluded_count:
            exclusions_report = {"rows": excluded_path, "summary": summary_path}
        return {
            "filtered": FilteredCustomData(frame=kept, excluded=excluded, original=original),
            "tables": tables,
            "prepared_paths": prepared_paths,
            "exclusions_report": exclusions_report,
            "relations_with_source": relations_with_source,
        }
    except (ImportError, KeyError, OSError, TypeError, ValueError) as exc:
        _emit(log_fn, f"Prepared inference TSV cache is invalid ({exc}); rebuilding")
        return None


def _emit(log_fn: Any | None, message: str) -> None:
    if log_fn is not None:
        log_fn(message)


@contextmanager
def _timed_step(log_fn: Any | None, message: str):
    start = time.perf_counter()
    _emit(log_fn, f"{message}...")
    try:
        yield
    finally:
        _emit(log_fn, f"{message} done in {time.perf_counter() - start:.1f}s")


def _iter_slices(row_count: int, chunk_size: int = PROGRESS_CHUNK_ROWS):
    for start in range(0, row_count, chunk_size):
        yield start, min(start + chunk_size, row_count)


def _progress(iterable: Any, *, total: int, desc: str, unit: str) -> Any:
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, total=total, desc=desc, unit=unit)


def _normalize_text_column(series: Any, column_name: str, log_fn: Any | None) -> Any:
    import pandas as pd

    chunks = []
    slices = list(_iter_slices(len(series)))
    with _timed_step(log_fn, f"Normalizing {column_name} text for {len(series):,} rows"):
        for start, end in _progress(slices, total=len(slices), desc=f"Normalize {column_name}", unit="chunk"):
            chunks.append(series.iloc[start:end].astype(str).str.strip())
    return pd.concat(chunks, ignore_index=True)


def _string_fullmatch(series: Any, pattern: str, desc: str, log_fn: Any | None) -> Any:
    import pandas as pd

    chunks = []
    slices = list(_iter_slices(len(series)))
    with _timed_step(log_fn, f"{desc} for {len(series):,} rows"):
        for start, end in _progress(slices, total=len(slices), desc=desc, unit="chunk"):
            chunks.append(series.iloc[start:end].str.fullmatch(pattern, na=False))
    return pd.concat(chunks, ignore_index=True)


def _string_len_gt(series: Any, limit: int, desc: str, log_fn: Any | None) -> Any:
    import pandas as pd

    chunks = []
    slices = list(_iter_slices(len(series)))
    with _timed_step(log_fn, f"{desc} for {len(series):,} rows"):
        for start, end in _progress(slices, total=len(slices), desc=desc, unit="chunk"):
            chunks.append(series.iloc[start:end].str.len() > limit)
    return pd.concat(chunks, ignore_index=True)


def _write_csv_with_progress(frame: Any, path: Path, *, sep: str, index: bool, log_fn: Any | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    chunks = list(_iter_slices(len(frame)))
    with _timed_step(log_fn, f"Writing {len(frame):,} rows to {path}"):
        for chunk_index, (start, end) in enumerate(
            _progress(chunks, total=len(chunks), desc=f"Write {path.name}", unit="chunk")
        ):
            frame.iloc[start:end].to_csv(
                path,
                sep=sep,
                index=index,
                header=chunk_index == 0,
                mode="w" if chunk_index == 0 else "a",
            )


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


def read_inference_pairs(
    path: Path,
    delimiter: str,
    has_header: bool,
    log_fn: Any | None = None,
) -> FilteredCustomData:
    import pandas as pd

    suffix = path.suffix.lower()
    _emit(log_fn, f"Inference input size: {path.stat().st_size / (1024 ** 3):.2f} GiB")
    with _timed_step(log_fn, f"Reading {suffix} inference input"):
        if suffix == ".csv":
            frame = pd.read_csv(path, sep=delimiter, header=0 if has_header else None)
        elif suffix == ".json":
            frame = pd.read_json(path)
        elif suffix == ".parquet":
            frame = pd.read_parquet(path)
        else:
            raise ValueError("Inference input must be a .csv, .json, or .parquet file.")
    _emit(log_fn, f"Loaded inference input: {len(frame):,} rows x {len(frame.columns):,} columns")

    if frame.empty:
        raise ValueError(f"No data rows were found in {path}")

    with _timed_step(log_fn, "Selecting smiles and sequence columns"):
        if suffix in {".json", ".parquet"} or has_header:
            lowered = {str(column).strip().lower(): column for column in frame.columns}
            expected_columns = {"smiles", "sequence"}
            if not expected_columns.issubset(lowered):
                raise ValueError("Inference input must contain columns named smiles and sequence.")
            frame = frame.rename(columns={lowered["smiles"]: "SMILES", lowered["sequence"]: "SEQ"})
            frame = frame[["SMILES", "SEQ"]]
        else:
            if frame.shape[1] != 2:
                raise ValueError(f"Expected 2 columns in {path}, found {frame.shape[1]}")
            frame = frame.iloc[:, :2]
            frame.columns = ["SMILES", "SEQ"]
        frame = frame.reset_index(drop=True)

    frame["SMILES"] = _normalize_text_column(frame["SMILES"], "SMILES", log_fn)
    frame["SEQ"] = _normalize_text_column(frame["SEQ"], "SEQ", log_fn)

    with _timed_step(log_fn, "Adding source row and placeholder labels"):
        frame = frame.reset_index(drop=True)
        frame.insert(0, "source_row", frame.index.astype(int))
        frame["label"] = 0

    empty_smiles = frame["SMILES"].eq("")
    _emit(log_fn, f"Validation empty_smiles: {int(empty_smiles.sum()):,} rows")
    empty_sequences = frame["SEQ"].eq("")
    _emit(log_fn, f"Validation empty_sequence: {int(empty_sequences.sum()):,} rows")
    invalid_sequences = ~_string_fullmatch(
        frame["SEQ"],
        r"[ACDEFGHIKLMNPQRSTVWY]+",
        "Validate canonical protein sequences",
        log_fn,
    )
    _emit(log_fn, f"Validation non_canonical_sequence: {int(invalid_sequences.sum()):,} rows")
    too_long_smiles = _string_len_gt(frame["SMILES"], 510, "Validate SMILES length", log_fn)
    _emit(log_fn, f"Validation smiles_too_long: {int(too_long_smiles.sum()):,} rows")
    too_long_sequences = _string_len_gt(frame["SEQ"], 700, "Validate sequence length", log_fn)
    _emit(log_fn, f"Validation sequence_too_long: {int(too_long_sequences.sum()):,} rows")

    with _timed_step(log_fn, "Building exclusion mask"):
        exclusion_reason = pd.Series("", index=frame.index, dtype="object")
        exclusion_reason = exclusion_reason.mask(empty_smiles, exclusion_reason.where(~empty_smiles, "empty_smiles"))
        exclusion_reason = exclusion_reason.mask(
            empty_sequences & exclusion_reason.eq(""),
            exclusion_reason.where(~(empty_sequences & exclusion_reason.eq("")), "empty_sequence"),
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

    with _timed_step(log_fn, "Materializing kept and excluded inference rows"):
        excluded = frame.loc[excluded_mask].copy()
        if not excluded.empty:
            excluded.insert(1, "reason", exclusion_reason.loc[excluded_mask].values)

        filtered = frame.loc[~excluded_mask].copy()
    _emit(log_fn, f"Inference validation kept {len(filtered):,} rows and excluded {len(excluded):,} rows")
    if filtered.empty:
        raise ValueError("All inference rows were filtered out. Check the exclusions report for details.")

    return FilteredCustomData(frame=filtered, excluded=excluded, original=frame.copy())


def save_exclusions_report(excluded: Any, output_dir: Path, log_fn: Any | None = None) -> dict[str, Path] | None:
    if excluded.empty:
        _emit(log_fn, "No excluded rows report to write")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    exclusions_path = output_dir / "excluded_rows.tsv"
    summary_path = output_dir / "excluded_summary.json"
    _write_csv_with_progress(excluded, exclusions_path, sep="\t", index=False, log_fn=log_fn)
    summary = {
        "excluded_rows": int(len(excluded)),
        "reasons": excluded["reason"].value_counts().sort_index().to_dict(),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _emit(log_fn, f"Wrote exclusions summary to {summary_path}")
    return {"rows": exclusions_path, "summary": summary_path}


def build_custom_tables(frame: Any, log_fn: Any | None = None) -> CustomDatasetTables:
    with _timed_step(log_fn, f"Deduplicating {len(frame):,} SMILES"):
        drug_table = frame[["SMILES"]].drop_duplicates().reset_index(drop=True)
        drug_table.insert(0, "drug_id", [f"drug_{index:06d}" for index in range(len(drug_table))])
    _emit(log_fn, f"Unique drugs: {len(drug_table):,}")

    with _timed_step(log_fn, f"Deduplicating {len(frame):,} protein sequences"):
        protein_table = frame[["SEQ"]].drop_duplicates().reset_index(drop=True)
        protein_table.insert(0, "protein_id", [f"protein_{index:06d}" for index in range(len(protein_table))])
    _emit(log_fn, f"Unique proteins: {len(protein_table):,}")

    with _timed_step(log_fn, "Building drug/protein lookup indices"):
        drug_lookup = drug_table.set_index("SMILES")["drug_id"]
        protein_lookup = protein_table.set_index("SEQ")["protein_id"]

    with _timed_step(log_fn, f"Mapping {len(frame):,} relations to deduplicated IDs"):
        relation_table = frame.copy()
        relation_table.insert(0, "Drug_ID", frame["SMILES"].map(drug_lookup))
        relation_table.insert(1, "Prot_ID", frame["SEQ"].map(protein_lookup))
        relation_table = relation_table[["Drug_ID", "Prot_ID", "label"]]

    with _timed_step(log_fn, "Finalizing custom tables"):
        x_drug = drug_table.rename(columns={"SMILES": "SMILES"}).set_index("drug_id")
        x_target = protein_table.rename(columns={"SEQ": "SEQ"}).set_index("protein_id")

    return CustomDatasetTables(X_drug=x_drug, X_target=x_target, DTI=relation_table)


def save_custom_tables(tables: CustomDatasetTables, output_dir: Path, log_fn: Any | None = None) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    drug_path = output_dir / "drug_table.tsv"
    protein_path = output_dir / "protein_table.tsv"
    relation_path = output_dir / "relation_table.tsv"
    _write_csv_with_progress(tables.X_drug.reset_index(names=["drug_id"]), drug_path, sep="\t", index=False, log_fn=log_fn)
    _write_csv_with_progress(
        tables.X_target.reset_index(names=["protein_id"]),
        protein_path,
        sep="\t",
        index=False,
        log_fn=log_fn,
    )
    _write_csv_with_progress(tables.DTI, relation_path, sep="\t", index=False, log_fn=log_fn)
    return {"drug_table": drug_path, "protein_table": protein_path, "relation_table": relation_path}


def custom_serializer_names(custom_data_path: Path, suffix: str) -> tuple[str, str]:
    stem = sanitize_slug(custom_data_path.stem)
    suffix_slug = sanitize_slug(suffix)
    return (
        f"custom_{stem}_{suffix_slug}_PubChem10M.pt",
        f"custom_{stem}_{suffix_slug}_ESM.pt",
    )


def ensure_prediction_runtime() -> None:
    """Fail before featurization when the Lightning runtime is incomplete."""
    try:
        import pytorch_lightning  # noqa: F401
    except ModuleNotFoundError as exc:
        if exc.name == "pkg_resources":
            raise RuntimeError(
                "The installed Lightning runtime requires pkg_resources. "
                "Install setuptools=68.0.0 or recreate the environment from "
                "environment.runpod.yaml before running prediction."
            ) from exc
        raise


def generate_custom_embeddings(
    cfg: Any,
    tables: CustomDatasetTables,
    serialized_dir: Path,
    drug_name: str,
    target_name: str,
    reuse_existing: bool,
    log_fn: Any | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Path]]:
    import hydra
    import pandas as pd
    import torch

    serialized_dir.mkdir(parents=True, exist_ok=True)
    drug_path = serialized_dir / drug_name
    target_path = serialized_dir / target_name

    if reuse_existing and drug_path.exists() and target_path.exists():
        _emit(log_fn, f"Reusing existing drug embeddings: {drug_path}")
        _emit(log_fn, f"Reusing existing protein embeddings: {target_path}")
        with _timed_step(log_fn, "Loading cached custom embeddings"):
            x_drug = torch.load(drug_path, map_location="cpu")
            x_target = torch.load(target_path, map_location="cpu")
        _emit(log_fn, f"Loaded embeddings: {len(x_drug):,} drugs, {len(x_target):,} proteins")
        return x_drug, x_target, {"drug_embedding": drug_path, "protein_embedding": target_path}

    relax_transformers_torch_load_guard()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _emit(log_fn, f"Generating custom embeddings on device: {device}")
    with _timed_step(log_fn, "Instantiating custom featurizers"):
        drug_featurizer = hydra.utils.instantiate(cfg["featurizer"]["drugfeaturizer"], device, _recursive_=False)
        prot_featurizer = hydra.utils.instantiate(cfg["featurizer"]["protfeaturizer"], device, _recursive_=False)

    with _timed_step(log_fn, f"Generating drug embeddings for {len(tables.X_drug):,} unique drugs"):
        drug_features = drug_featurizer.get_representations(tables.X_drug.SMILES.values)
    with _timed_step(log_fn, f"Generating protein embeddings for {len(tables.X_target):,} unique proteins"):
        target_features = prot_featurizer.get_representations(tables.X_target.SEQ.values)
    with _timed_step(log_fn, "Materializing embedding DataFrames"):
        x_drug = pd.DataFrame(drug_features, index=tables.X_drug.index)
        x_target = pd.DataFrame(target_features, index=tables.X_target.index)
    with _timed_step(log_fn, f"Saving drug embeddings to {drug_path}"):
        torch.save(x_drug, drug_path)
    with _timed_step(log_fn, f"Saving protein embeddings to {target_path}"):
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


def save_inference_export(
    filtered_custom_data: FilteredCustomData,
    prediction_rows: Any,
    output_dir: Path,
    input_path: Path,
    delimiter: str,
    output_csv: Path | None = None,
    log_fn: Any | None = None,
) -> dict[str, Path]:
    import numpy as np
    import pandas as pd

    output_dir.mkdir(parents=True, exist_ok=True)

    with _timed_step(log_fn, f"Building inference export frame for {len(filtered_custom_data.original):,} input rows"):
        export_frame = filtered_custom_data.original.copy()
        export_frame = export_frame.rename(columns={"SMILES": "smiles", "SEQ": "sequence"})
        export_frame["label"] = pd.Series([pd.NA] * len(export_frame), dtype="Int64")
        export_frame["probability"] = np.nan

    with _timed_step(log_fn, f"Indexing {len(prediction_rows):,} prediction rows"):
        prediction_indexed = prediction_rows.set_index("source_row")
    with _timed_step(log_fn, "Mapping predictions onto original input rows"):
        prediction_mask = export_frame["source_row"].isin(prediction_indexed.index)
        export_frame.loc[prediction_mask, "label"] = export_frame.loc[prediction_mask, "source_row"].map(
            prediction_indexed["predicted_label"]
        )
        export_frame.loc[prediction_mask, "probability"] = export_frame.loc[prediction_mask, "source_row"].map(
            prediction_indexed["probability_active"]
        )
    _emit(log_fn, f"Mapped predictions for {int(prediction_mask.sum()):,} rows")

    with _timed_step(log_fn, "Selecting inference export columns"):
        export_frame = export_frame.drop(columns=["source_row"])
        export_frame = export_frame[["smiles", "sequence", "label", "probability"]]

    csv_path = output_dir / "inference_predictions.csv" if output_csv is None else output_csv
    _write_csv_with_progress(export_frame, csv_path, sep=delimiter, index=False, log_fn=log_fn)

    outputs = {"csv": csv_path}
    if input_path.suffix.lower() in {".json", ".parquet"} and output_csv is None:
        json_path = output_dir / "inference_predictions.json"
        with _timed_step(log_fn, f"Writing inference JSON export to {json_path}"):
            json_path.write_text(export_frame.to_json(orient="records", indent=2), encoding="utf-8")
        outputs["json"] = json_path
    elif input_path.suffix.lower() in {".json", ".parquet"}:
        _emit(log_fn, "Skipping JSON sidecar because --output-csv was provided")

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
    progress_desc: str = "Predicting",
    log_fn: Any | None = None,
) -> pd.DataFrame:
    import pandas as pd
    import torch

    from datamodule.dataloader_GAT import MyDataset
    from module.cognn_cross import Net

    if log_fn is not None:
        log_fn(f"Loading checkpoint: {checkpoint_path}")
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
    if log_fn is not None:
        log_fn(f"Using device: {device}")

    test_table = dataset["test"].reset_index(drop=True)
    prediction_dataset = MyDataset(dataset["X_drug"], dataset["X_target"], test_table)
    dataloader = torch.utils.data.DataLoader(
        prediction_dataset,
        batch_size=cfg_dict["datamodule"]["dm_cfg"]["batch_size"],
        shuffle=False,
        num_workers=0,
    )
    if log_fn is not None:
        log_fn(
            f"Scoring {len(prediction_dataset)} rows in {len(dataloader)} batches "
            f"(batch_size={cfg_dict['datamodule']['dm_cfg']['batch_size']})"
        )

    probabilities: list[float] = []
    true_labels: list[int] = []

    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = None

    batches = dataloader
    if tqdm is not None:
        batches = tqdm(dataloader, desc=progress_desc, unit="batch")

    with torch.no_grad():
        for batch in batches:
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

    if log_fn is not None:
        log_fn(f"Finished scoring {len(probabilities)} rows")

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
