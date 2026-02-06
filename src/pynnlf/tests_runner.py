#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
from datetime import datetime
import shutil
import pandas as pd

from .yamlio import load_yaml
from .runner import run_batch
from .engine import compute_exp_no  # or keep local helper if you moved it
import re
from .recap_experiments import recap_experiments


def _workspace_root_from_spec(spec_path: Path) -> Path:
    """
    Infer workspace root from a spec path.

    Args:
        spec_path (Path): <workspace>/specs/tests_*.yaml

    Returns:
        Path: workspace root
    """
    return spec_path.parent.parent


def _find_latest_experiment_dir(output_dir: Path) -> Path:
    """
    Find the most recently created experiment directory matching E00001_* pattern.

    Args:
        output_dir (Path): <workspace>/experiment_result

    Returns:
        Path: latest experiment directory
    """
    pat = re.compile(r"^E(\d{5})_")
    dirs = []
    for p in output_dir.iterdir():
        if p.is_dir() and pat.match(p.name):
            dirs.append(p)
    if not dirs:
        raise FileNotFoundError(f"No experiment folders found in {output_dir}")
    # newest by modification time
    return max(dirs, key=lambda x: x.stat().st_mtime)


def _load_benchmark_csv(workspace_root: Path, benchmark_rel: str) -> pd.DataFrame:
    """
    Load benchmark CSV (index-style).

    Args:
        workspace_root (Path): workspace root
        benchmark_rel (str): e.g. "specs/testing_benchmark.csv"

    Returns:
        pd.DataFrame: indexed by metric_id with min/max columns
    """
    p = workspace_root / benchmark_rel
    df = pd.read_csv(p)
    df = df.rename(columns={df.columns[0]: "metric_id"}) if df.columns[0] != "metric_id" else df
    df = df.set_index("metric_id")
    return df


def _write_report(report_path: Path, report_rows: list[dict]) -> None:
    """
    Write report CSV in your existing format.

    Args:
        report_path (Path): output file path
        report_rows (list[dict]): list of report rows

    Returns:
        None
    """
    expected_cols = ["metric_id", "min acceptable value", "max acceptable value", "value, test", "test_result"]

    # If nothing matched, write an empty report with headers (no crash)
    if not report_rows:
        out = pd.DataFrame(columns=["metric_id","min acceptable value","max acceptable value","value, test","test_result"])
        out = out.set_index("metric_id")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(report_path, index=True)
        return

    out = pd.DataFrame(report_rows)

    if "metric_id" not in out.columns:
        raise KeyError(f"'metric_id' missing in report rows. Keys seen: {list(out.columns)}")

    out = out.set_index("metric_id")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(report_path, index=True)


def _run_recap_smoke(output_dir: Path) -> list[str]:
    warnings: list[str] = []
    temp_root = output_dir / "_recap_smoke"

    if temp_root.exists():
        shutil.rmtree(temp_root, ignore_errors=True)
    temp_root.mkdir(parents=True, exist_ok=True)

    def _make_folder(name: str) -> Path:
        p = temp_root / name
        p.mkdir(parents=True, exist_ok=False)
        return p

    folder_1 = _make_folder("E00001_smoke")
    folder_2 = _make_folder("E00002_smoke")
    folder_3 = _make_folder("E00003_smoke")
    folder_4 = _make_folder("E00004_smoke")

    (folder_1 / "foo_experiment_result.csv").write_text(
        "col1,col2\n1,one\n2,two\n",
        encoding="utf-8",
    )

    (folder_3 / "bad_experiment_result.csv").write_text(
        'col1,col2\n"unterminated,1\n',
        encoding="utf-8",
    )

    (folder_4 / "bar_experiment_result.csv").write_text(
        "col1,col2\n4,four\n5,five\n",
        encoding="utf-8",
    )

    out_path = temp_root / "a1_experiment_result.csv"
    df = recap_experiments(temp_root, out_path, return_df=True, include_experiment_folder=True)

    expected_folders = ["E00001_smoke", "E00004_smoke"]
    actual_folders = df["experiment_folder"].tolist() if "experiment_folder" in df.columns else []
    if actual_folders != expected_folders:
        warnings.append(
            f"recap ordering mismatch: expected {expected_folders} got {actual_folders}"
        )

    expected_col1 = [1, 4]
    actual_col1 = df["col1"].tolist() if "col1" in df.columns else []
    if actual_col1 != expected_col1:
        warnings.append(
            f"recap nrows=1 mismatch: expected {expected_col1} got {actual_col1}"
        )

    if not out_path.exists():
        warnings.append(f"recap output missing: {out_path}")

    shutil.rmtree(temp_root, ignore_errors=True)
    return warnings


def run_tests(spec_path: str | Path) -> Path:
    """
    Run regression tests described by specs/tests_ci.yaml or specs/tests_full.yaml.

    Pure warning mode:
        - Always completes and writes a report.
        - Never raises / never fails CI.

    Args:
        spec_path (str | Path): path to <workspace>/specs/tests_*.yaml

    Returns:
        Path: path to generated report CSV
    """
    spec_path = Path(spec_path)
    ws = _workspace_root_from_spec(spec_path)

    tests = load_yaml(spec_path)
    cfg = load_yaml(ws / "specs" / "pynnlf_config.yaml")

    output_dir = ws / cfg["paths"]["output_dir"]

    # 1) Run experiments (reuse batch runner)
    # Build a temporary batch spec in-memory and write to a temp yaml in specs/
    batch_spec = {
        "datasets": tests["datasets"],
        "forecast_horizons": tests["forecast_horizons"],
        "model_and_hp": tests["model_and_hp"],
    }
    tmp_batch = ws / "specs" / "_tmp_tests_batch.yaml"
    tmp_batch.write_text(
        "\n".join([
            f"datasets: {batch_spec['datasets']}",
            f"forecast_horizons: {batch_spec['forecast_horizons']}",
            "model_and_hp:",
            *[f"  - [{m}, {hp}]" for m, hp in batch_spec["model_and_hp"]],
        ]),
        encoding="utf-8"
    )

    run_batch(tmp_batch)

    recap_warnings = _run_recap_smoke(output_dir)
    for w in recap_warnings:
        print(f"[PyNNLF recap] WARNING: {w}")

    # Also produce a real recap summary for the experiment_result root.
    try:
        recap_experiments(output_dir)
    except Exception as exc:
        print(f"[PyNNLF recap] WARNING: failed to write recap summary: {exc}")

    # 2) Load benchmark
    bench = _load_benchmark_csv(ws, tests["benchmark_csv"])

    # 3) For each newly created experiment folder, read a3 and compare metrics
    # NOTE: simplest approach: compare using latest experiment folder repeatedly.
    # Since run_batch creates multiple experiment folders, we scan all new ones.
    pat = re.compile(r"^E(\d{5})_")
    exp_dirs = sorted([p for p in output_dir.iterdir() if p.is_dir() and pat.match(p.name)],
                      key=lambda x: x.stat().st_mtime)

    report_rows = []

    for exp_dir in exp_dirs:
        # a3 filename pattern: <E00001>_a3_cross_validation_result.csv
        # we detect it by suffix
        a3_files = list(exp_dir.glob("*_a3_cross_validation_result.csv"))
        if not a3_files:
            continue
        a3_path = a3_files[0]

        a3 = pd.read_csv(a3_path, index_col=0)
        # expecting mean/stddev rows exist
        if "mean" not in a3.index or "stddev" not in a3.index:
            continue

        parts = exp_dir.name.split("_")

        # folder pattern: E00001_YYMMDD_ds0_fh30_<model_name...>_<hp_no>
        hp_no = parts[-1]
        model_name = "_".join(parts[4:-1])  # join all tokens after fhXX up to hpXX

        # Compare each metric in a3 columns for mean and stddev
        for col in a3.columns:
            for stat in ("mean", "stddev"):
                metric_id = f"{model_name}_{col}_{stat}"
                if metric_id not in bench.index:
                    continue

                v = float(a3.loc[stat, col])
                vmin = float(bench.loc[metric_id, "min acceptable value"])
                vmax = float(bench.loc[metric_id, "max acceptable value"])

                eps = float(tests.get("tolerance", 0.005))  # default: 2dp rounding half-step
                passed = (v >= (vmin - eps)) and (v <= (vmax + eps))
                report_rows.append({
                    "metric_id": metric_id,
                    "min acceptable value": vmin,
                    "max acceptable value": vmax,
                    "value, test": v,
                    "test_result": "pass" if passed else "fail",
                })

    # 4) Write report (pure warning)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / "Archive" / "Testing Result" / f"test_result_{ts}.csv"
    
    if not report_rows:
        print("[PyNNLF tests] WARNING: No metrics matched benchmark CSV keys. Report will be empty.")
    
    _write_report(report_path, report_rows)

    # Print summary, but do not raise
    n_fail = sum(1 for r in report_rows if r["test_result"] == "fail")
    n_pass = sum(1 for r in report_rows if r["test_result"] == "pass")
    print(f"[PyNNLF tests] pass={n_pass} fail={n_fail} report={report_path}")

    # cleanup temp file
    try:
        tmp_batch.unlink()
    except Exception:
        pass

    return report_path