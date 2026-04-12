#!/usr/bin/env python3
"""
Enumerate and optionally execute MORPH finetune jobs for Sweep A / B.

Usage (from repo root)::

    PYTHONPATH=code python -m morph_wrap.run_sweep --sweep A --dry-run
    PYTHONPATH=code python -m morph_wrap.run_sweep --sweep A --execute --morph-root MORPH

HPC / job array: write a manifest, then one Slurm array task per row::

    PYTHONPATH=code python -m morph_wrap.run_sweep --sweep A --dry-run --manifest-csv out/sweep_A_manifest.csv
    PYTHONPATH=code python -m morph_wrap.run_sweep --execute --manifest-csv out/sweep_A_manifest.csv \\
        --execute-job-index $SLURM_ARRAY_TASK_ID --array-base 1

See ``scripts/hpc/`` for Slurm examples and optional ``module load`` helper.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

from morph_wrap.device_resolve import (
    format_device_resolution_log,
    resolve_training_device_index,
)
from morph_wrap.hpc_util import append_failure_log, append_summary_line, load_manifest_entries
from morph_wrap.morph_cli import finetune_argv, finetune_env
from morph_wrap.sweep_config import (
    DATASET_ORDER,
    DatasetKey,
    SweepJob,
    filter_jobs_by_models,
    fm_checkpoint_for,
    lr_for,
    staged_models,
    sweep_a_jobs,
    sweep_b_jobs,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _write_manifest_csv(
    path: Path,
    jobs: list[SweepJob],
    commands: list[str],
    checkpoints: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
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
            ]
        )
        for job, cmd, ck in zip(jobs, commands, checkpoints):
            w.writerow(
                [
                    job.sweep,
                    job.combo_index,
                    job.combo_total,
                    job.run_id,
                    job.dataset,
                    job.model,
                    job.morph_ft_dataset,
                    job.morph_model_size,
                    job.train_frac,
                    job.epochs,
                    job.context_frames,
                    job.n_traj,
                    lr_for(job.dataset, job.model, job.context_frames),
                    ck,
                    cmd,
                ]
            )


def _load_context_json(path: Path) -> dict[DatasetKey, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[DatasetKey, int] = {}
    for k in DATASET_ORDER:
        if k not in data:
            raise KeyError(f"context json missing key {k!r}")
        out[k] = int(data[k])
    return out


def _run_one_finetune(
    job: SweepJob,
    argv: list[str],
    *,
    morph_root: Path,
    out_dir: Path,
    pilot_lr_override: str | None,
) -> int:
    env = finetune_env(job, out_dir, pilot_lr_override=pilot_lr_override)
    print(f"\n>>> EXEC {job.run_id}", flush=True)
    # Inherit stdout/stderr so batch systems capture logs (redirect in sbatch/qsub).
    proc = subprocess.run(
        [sys.executable, *argv],
        cwd=os.fspath(morph_root),
        env=env,
    )
    return proc.returncode


def _execute_job_list(
    jobs: list[SweepJob],
    argv_list: list[list[str]],
    *,
    morph_root: Path,
    out_dir: Path,
    pilot_lr_overrides: list[str | None],
    failure_log: Path,
    continue_on_failure: bool,
    summary_log: Path | None,
) -> int:
    failures: list[tuple[SweepJob, int]] = []
    for job, argv, plr in zip(jobs, argv_list, pilot_lr_overrides):
        code = _run_one_finetune(
            job,
            argv,
            morph_root=morph_root,
            out_dir=out_dir,
            pilot_lr_override=plr,
        )
        if code != 0:
            cmd = " ".join(argv[:3]) + " ..."
            append_failure_log(
                failure_log,
                run_id=job.run_id,
                returncode=code,
                sweep=job.sweep,
                combo_index=job.combo_index,
                message="finetune_MORPH.py nonzero exit",
                command=cmd,
            )
            print(
                f"FAILED {job.run_id} exit={code} (logged to {failure_log})",
                file=sys.stderr,
                flush=True,
            )
            failures.append((job, code))
            if not continue_on_failure:
                if summary_log:
                    append_summary_line(
                        summary_log,
                        f"ABORT after failure n_failures={len(failures)} last={job.run_id}",
                    )
                return code

    if summary_log:
        append_summary_line(
            summary_log,
            f"batch_done jobs={len(jobs)} failures={len(failures)}",
        )

    if failures:
        print(
            f"Completed with {len(failures)} failure(s); see {failure_log}",
            file=sys.stderr,
            flush=True,
        )
        return 1
    return 0


def _resolve_array_row_index(task_id: int, array_base: int) -> int:
    """``array_base`` 1 -> Slurm 1..N maps to row 0..N-1; ``array_base`` 0 -> task id is row index."""
    if array_base == 1:
        return task_id - 1
    if array_base == 0:
        return task_id
    raise ValueError(f"array_base must be 0 or 1, got {array_base}")


def main() -> int:
    root = _repo_root()
    p = argparse.ArgumentParser(description="MORPH surrogate parameter sweeps")
    p.add_argument(
        "--sweep",
        choices=("A", "B"),
        default=None,
        help="Which factorial to expand (not required for manifest-only single-job mode)",
    )
    p.add_argument("--dry-run", action="store_true", help="Print commands only")
    p.add_argument("--execute", action="store_true", help="Run finetune_MORPH.py")
    p.add_argument("--limit", type=int, default=0, help="Max jobs (0 = no limit)")
    p.add_argument("--morph-root", type=Path, default=root / "MORPH")
    p.add_argument("--dataset-root", type=Path, default=root / "MORPH")
    p.add_argument(
        "--fm-checkpoint",
        type=str,
        default="",
        help="Optional: use this FM checkpoint basename for every job (overrides per-size table)",
    )
    p.add_argument(
        "--context-json",
        type=Path,
        default=root / "out" / "sweep_b_context.json",
        help="Sweep B: per-dataset winning context_frames (from pick_context)",
    )
    p.add_argument(
        "--include-large",
        action="store_true",
        help="Sweep B: include large (L) model jobs (default: tiny+small only)",
    )
    p.add_argument("--manifest-csv", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=root / "out")
    p.add_argument(
        "--device-index",
        type=str,
        default="auto",
        metavar="auto|INT",
        help="MORPH --device_idx: 'auto' = MORPH_DEVICE_IDX if set, else torch.cuda.current_device() "
        "(see torch.cuda.device_count()); else integer index",
    )
    p.add_argument(
        "--quiet-device",
        action="store_true",
        help="Suppress stderr line showing resolved --device_idx",
    )
    p.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="On --execute batch: keep going after a failed job; exit 1 if any failed",
    )
    p.add_argument(
        "--failure-log",
        type=Path,
        default=None,
        help="Append JSON lines per failure (default: <out-dir>/sweep_failures.jsonl)",
    )
    p.add_argument(
        "--summary-log",
        type=Path,
        default=None,
        help="Append brief UTC lines for batch start/end (optional)",
    )
    p.add_argument(
        "--execute-job-index",
        type=int,
        default=None,
        help="Run exactly one row from --manifest-csv (1-based if --array-base 1)",
    )
    p.add_argument(
        "--use-slurm-array-task-id",
        action="store_true",
        help="Use env SLURM_ARRAY_TASK_ID as job index (with --array-base)",
    )
    p.add_argument(
        "--array-base",
        type=int,
        choices=(0, 1),
        default=1,
        help="1: SLURM_ARRAY_TASK_ID is 1..N; 0: task id is 0..N-1",
    )

    args = p.parse_args()

    di_raw = args.device_index.strip().lower()
    if di_raw == "auto":
        explicit_dev: int | None = None
    else:
        try:
            explicit_dev = int(args.device_index)
        except ValueError:
            print(
                f"Invalid --device-index {args.device_index!r}; use 'auto' or an integer.",
                file=sys.stderr,
            )
            return 2
        if explicit_dev < 0:
            print("--device-index must be >= 0", file=sys.stderr)
            return 2

    device_idx, dev_info = resolve_training_device_index(explicit_dev)
    if not args.quiet_device:
        print(
            format_device_resolution_log(
                device_idx,
                dev_info,
                explicit_cli=explicit_dev is not None,
            ),
            file=sys.stderr,
            flush=True,
        )

    morph_root = args.morph_root.resolve()
    out_dir = args.out_dir.resolve()
    failure_log = (args.failure_log or (out_dir / "sweep_failures.jsonl")).resolve()
    summary_log_path = args.summary_log.resolve() if args.summary_log else None

    finetune_py = morph_root / "scripts" / "finetune_MORPH.py"
    if not finetune_py.is_file():
        print(f"Missing {finetune_py}", file=sys.stderr)
        return 1

    # --- Single job from manifest (HPC array) ---
    use_manifest_only = (
        args.execute_job_index is not None or args.use_slurm_array_task_id
    )
    if use_manifest_only:
        if args.manifest_csv is None:
            print(
                "Manifest mode requires --manifest-csv (path to CSV from a prior --dry-run).",
                file=sys.stderr,
            )
            return 1
        manifest = args.manifest_csv.resolve()
        if not manifest.is_file():
            print(f"Manifest not found: {manifest}", file=sys.stderr)
            return 1

        if args.use_slurm_array_task_id:
            raw = os.environ.get("SLURM_ARRAY_TASK_ID")
            if raw is None:
                print(
                    "SLURM_ARRAY_TASK_ID is unset (submit with a job array or pass "
                    "--execute-job-index).",
                    file=sys.stderr,
                )
                return 1
            task_id = int(raw)
        else:
            task_id = args.execute_job_index  # type: ignore[assignment]

        try:
            entries = load_manifest_entries(manifest)
        except (ValueError, KeyError) as e:
            print(f"Manifest error: {e}", file=sys.stderr)
            return 1

        row_index = _resolve_array_row_index(task_id, args.array_base)
        if row_index < 0 or row_index >= len(entries):
            print(
                f"Job index out of range: resolved row {row_index} len={len(entries)} "
                f"(task_id={task_id} array_base={args.array_base})",
                file=sys.stderr,
            )
            return 1

        entry = entries[row_index]
        job = entry.job
        argv = finetune_argv(
            job,
            morph_root=morph_root,
            dataset_root=args.dataset_root.resolve(),
            fm_checkpoint=entry.fm_checkpoint,
            device_idx=device_idx,
        )
        cmd_str = " ".join([sys.executable, *argv])
        print(f"[manifest row {row_index + 1}/{len(entries)}] {job.run_id}")
        print(cmd_str)

        if args.dry_run and not args.execute:
            return 0

        if args.execute:
            if summary_log_path:
                append_summary_line(
                    summary_log_path,
                    f"single_start run_id={job.run_id} row={row_index}",
                )
            code = _run_one_finetune(
                job,
                argv,
                morph_root=morph_root,
                out_dir=out_dir,
                pilot_lr_override=entry.pilot_lr or None,
            )
            if code != 0:
                append_failure_log(
                    failure_log,
                    run_id=job.run_id,
                    returncode=code,
                    sweep=job.sweep,
                    combo_index=job.combo_index,
                    message="manifest single-job finetune failed",
                    command=cmd_str[:500],
                )
                print(
                    f"FAILED {job.run_id} exit={code} (logged to {failure_log})",
                    file=sys.stderr,
                    flush=True,
                )
            if summary_log_path:
                append_summary_line(
                    summary_log_path,
                    f"single_end run_id={job.run_id} exit={code}",
                )
            return code

        return 0

    # --- Factorial sweep ---
    if args.sweep is None:
        print("Provide --sweep A|B or use --manifest-csv with --execute-job-index.", file=sys.stderr)
        return 1

    if args.sweep == "A":
        jobs = sweep_a_jobs()
    else:
        ctx = _load_context_json(args.context_json)
        jobs = sweep_b_jobs(ctx)
        if not args.include_large:
            jobs = filter_jobs_by_models(jobs, staged_models("small"))

    if args.limit and args.limit > 0:
        jobs = jobs[: args.limit]

    if not jobs:
        print("No jobs to run.", file=sys.stderr)
        return 1

    manifest = args.manifest_csv or (
        args.out_dir / f"sweep_{args.sweep}_manifest_{len(jobs)}.csv"
    )

    ckpt_override = args.fm_checkpoint.strip() or None

    cmd_strings: list[str] = []
    ckpts: list[str] = []
    argv_rows: list[list[str]] = []
    for job in jobs:
        ck = fm_checkpoint_for(job.model, ckpt_override)
        ckpts.append(ck)
        argv = finetune_argv(
            job,
            morph_root=morph_root,
            dataset_root=args.dataset_root.resolve(),
            fm_checkpoint=ck,
            device_idx=device_idx,
        )
        argv_rows.append(argv)
        cmd_strings.append(" ".join([sys.executable, *argv]))

    _write_manifest_csv(manifest.resolve(), jobs, cmd_strings, ckpts)
    print(f"Wrote manifest: {manifest}")

    for job, cmd in zip(jobs, cmd_strings):
        print(f"\n[{job.combo_index}/{job.combo_total}] {job.run_id}")
        print(cmd)

    if args.execute:
        if summary_log_path:
            append_summary_line(
                summary_log_path,
                f"batch_start sweep={args.sweep} n_jobs={len(jobs)}",
            )
        pilot_overrides: list[str | None] = [None] * len(jobs)
        return _execute_job_list(
            jobs,
            argv_rows,
            morph_root=morph_root,
            out_dir=out_dir,
            pilot_lr_overrides=pilot_overrides,
            failure_log=failure_log,
            continue_on_failure=args.continue_on_failure,
            summary_log=summary_log_path,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
