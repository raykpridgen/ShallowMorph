"""HPC helpers: manifest parsing, structured failure logging."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, cast

from morph_wrap.sweep_config import DatasetKey, ModelKey, SweepJob, SweepName

MANIFEST_COLUMNS = {
    "sweep",
    "combo_index",
    "combo_total",
    "run_id",
    "dataset",
    "model",
    "morph_ft_dataset",
    "morph_model_size",
    "train_frac",
    "epochs",
    "context_frames",
    "n_traj",
    "lr_pilot_table",
    "fm_checkpoint_basename",
    "command",
}


@dataclass(frozen=True)
class ManifestEntry:
    """One row of a sweep manifest CSV (see ``run_sweep`` writer)."""

    job: SweepJob
    fm_checkpoint: str
    pilot_lr: str


def _parse_manifest_row(row: dict[str, str]) -> ManifestEntry:
    missing = MANIFEST_COLUMNS - row.keys()
    if missing:
        raise ValueError(f"Manifest row missing columns: {sorted(missing)}")

    ds = row["dataset"].strip()
    mk = row["model"].strip()
    sw = row["sweep"].strip()
    if ds not in ("BE1D", "SW2D", "CFD3D"):
        raise ValueError(f"Invalid dataset {ds!r} in manifest")
    if mk not in ("tiny", "small", "large"):
        raise ValueError(f"Invalid model {mk!r} in manifest")
    if sw not in ("A", "B"):
        raise ValueError(f"Invalid sweep {sw!r} in manifest")
    ds_k = cast(DatasetKey, ds)
    mk_k = cast(ModelKey, mk)
    sw_k = cast(SweepName, sw)

    job = SweepJob(
        sweep=sw_k,
        combo_index=int(row["combo_index"]),
        combo_total=int(row["combo_total"]),
        dataset=ds_k,
        model=mk_k,
        train_frac=int(row["train_frac"]),
        epochs=int(row["epochs"]),
        context_frames=int(row["context_frames"]),
        run_id=row["run_id"].strip(),
    )
    return ManifestEntry(
        job=job,
        fm_checkpoint=row["fm_checkpoint_basename"].strip(),
        pilot_lr=row["lr_pilot_table"].strip(),
    )


def load_manifest_entries(manifest_path: Path) -> List[ManifestEntry]:
    """Load all jobs from a manifest written by ``run_sweep --dry-run``."""
    with manifest_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty or invalid CSV: {manifest_path}")
        cols = set(h.strip() for h in reader.fieldnames)
        need = MANIFEST_COLUMNS
        if not need <= cols:
            raise ValueError(
                f"Manifest {manifest_path} missing columns {sorted(need - cols)}; "
                f"found {sorted(cols)}"
            )
        return [_parse_manifest_row({k: (row.get(k) or "") for k in need}) for row in reader]


def append_failure_log(
    path: Path,
    *,
    run_id: str,
    returncode: int,
    sweep: str,
    combo_index: int,
    message: str = "",
    command: str = "",
) -> None:
    """Append one JSON line (easy to grep / parse on HPC)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "returncode": returncode,
        "sweep": sweep,
        "combo_index": combo_index,
    }
    if message:
        record["message"] = message
    if command:
        record["command"] = command
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_summary_line(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    with path.open("a", encoding="utf-8") as f:
        f.write(f"{ts}\t{text}\n")
