from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import nbformat as nbf
import numpy as np
import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_DIR = Path(__file__).resolve().parents[1]

RAW_BESS_DIR = Path(
    r"C:\Users\z5404477\OneDrive - UNSW\H0424909\04_Workspace\2. WIP\data\1. raw\Solar Analytics Data from CICCADA\bess_data"
)
SOURCE_NOTEBOOK_DIR = (
    REPO_ROOT
    / "ignored_from_git"
    / "notebook_process_data"
    / "Process CICCADA Solar Analytics BESS data"
)

CLEANED_SA_BESS_DIR = Path(
    r"C:\Users\z5404477\OneDrive - UNSW\H0424909\04_Workspace\2. WIP\data\3. cleaned\SA BESS"
)
DIAGNOSTICS_DIR = CLEANED_SA_BESS_DIR / "diagnostics"
INTERMEDIATE_DIR = CLEANED_SA_BESS_DIR / "intermediate"
PROCESSED_DIR = CLEANED_SA_BESS_DIR / "processed"
RESULTS_DIR = WORKSPACE_DIR / "results"
NOTEBOOK_DIR = WORKSPACE_DIR / "notebooks"

SITE_META_PATH = RAW_BESS_DIR / "site_meta_data.csv"
CIRCUIT_META_PATH = RAW_BESS_DIR / "circuit_meta_data.csv"
REPORT_PATH = DIAGNOSTICS_DIR / "sa_bess_signal_diagnostics_report.md"

CURRENT_PROFILE = "current_polarity_adjusted"
N_HOUSEHOLDS = 100
N_DAYS = 365
FIVE_MINUTES_PER_DAY = 288
EXPECTED_5MIN_ROWS = N_DAYS * FIVE_MINUTES_PER_DAY
NEGATIVE_LOAD_TOLERANCE_KW = -0.05
SELECTION_MISSINGNESS_THRESHOLDS_PCT = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0]
SELECTION_PRIOR_FILL_BUFFER_DAYS = 10
FIXED_AEST_OFFSET_HOURS = 10

REQUIRED_CIRCUIT_TYPES = ["ac_load_net", "pv_site_net", "battery_storage"]
DIRECT_CIRCUIT_TYPES = ["ac_load", "pv_site"]
ALL_DIAGNOSTIC_CIRCUIT_TYPES = REQUIRED_CIRCUIT_TYPES + DIRECT_CIRCUIT_TYPES

TARGET_COLUMNS = [
    "underlying_load_kW",
    "net_load_with_pv_kW",
    "net_load_with_pv_and_battery_kW",
]

DATASET_SPECS = {
    "ds16": {
        "filename": "ds16_sa_bess_underlying_load_30min.csv",
        "source_column": "underlying_load_kW",
        "label": "underlying_load",
        "description": "Solar Analytics CICCADA BESS aggregate of 100 households, underlying load, 30-minute mean power.",
    },
    "ds17": {
        "filename": "ds17_sa_bess_net_load_with_pv_30min.csv",
        "source_column": "net_load_with_pv_kW",
        "label": "net_load_with_pv",
        "description": "Solar Analytics CICCADA BESS aggregate of 100 households, net load with PV, 30-minute mean power.",
    },
    "ds18": {
        "filename": "ds18_sa_bess_net_load_with_pv_battery_30min.csv",
        "source_column": "net_load_with_pv_and_battery_kW",
        "label": "net_load_with_pv_battery",
        "description": "Solar Analytics CICCADA BESS aggregate of 100 households, net load with PV and battery, 30-minute mean power.",
    },
}

MODEL_AND_HP = [
    ["m17", "hp1"],
    ["m8", "hp1"],
    ["m6", "hp1"],
    ["m10", "hp1"],
    ["m7", "hp1"],
    ["m1", "hp1"],
    ["m3", "hp1"],
    ["m9", "hp3"],
    ["m13", "hp2"],
    ["m2", "hp2"],
    ["m16", "hp1"],
    ["m4", "hp1"],
]


@dataclass(frozen=True)
class FormulaProfile:
    """Formula definition used for diagnostics and selected-sample processing."""

    profile_id: str
    label: str
    basis: str
    battery_positive_meaning: str
    notes: str


FORMULA_PROFILES = {
    CURRENT_PROFILE: FormulaProfile(
        profile_id=CURRENT_PROFILE,
        label="Documented polarity-adjusted hypothesis",
        basis="adjusted_power",
        battery_positive_meaning="battery net charge",
        notes=(
            "Uses power multiplied by circuit_polarity. Positive battery_storage is treated as charging; "
            "ac_load_net is treated as net load after PV and battery."
        ),
    ),
    "battery_discharge_positive_adjusted": FormulaProfile(
        profile_id="battery_discharge_positive_adjusted",
        label="Battery-discharge-positive alternative",
        basis="adjusted_power",
        battery_positive_meaning="battery net discharge",
        notes=(
            "Uses power multiplied by circuit_polarity, but treats positive battery_storage as discharging."
        ),
    ),
    "raw_processing_like": FormulaProfile(
        profile_id="raw_processing_like",
        label="Raw-power processing-notebook alternative",
        basis="raw_power",
        battery_positive_meaning="legacy processing-derived",
        notes=(
            "Uses raw power and the earlier processing notebook formulas. This is included mainly as a falsification check."
        ),
    ),
}


def ensure_directories() -> None:
    """Create output folders used by the publication workflow."""

    for path in [
        DIAGNOSTICS_DIR,
        INTERMEDIATE_DIR,
        PROCESSED_DIR,
        RESULTS_DIR,
        NOTEBOOK_DIR,
        WORKSPACE_DIR / "data",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def parquet_paths() -> list[Path]:
    """Return daily raw parquet paths in stable processing order."""

    paths = sorted(RAW_BESS_DIR.rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No raw parquet files found under {RAW_BESS_DIR}")
    return paths


def load_metadata() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load site and circuit metadata from the CICCADA Solar Analytics BESS dump."""

    site_meta = pd.read_csv(SITE_META_PATH)
    circuit_meta = pd.read_csv(CIRCUIT_META_PATH)
    site_meta["monitoring_start"] = pd.to_datetime(site_meta["monitoring_start"], errors="coerce")
    return site_meta, circuit_meta


def build_circuit_lookup(
    circuit_meta: pd.DataFrame,
    circuit_types: Iterable[str],
    site_ids: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Create the circuit-to-site lookup used to join raw parquet rows to site metadata."""

    lookup = circuit_meta.loc[circuit_meta["circuit_type"].isin(list(circuit_types))].copy()
    if site_ids is not None:
        site_id_set = {int(site_id) for site_id in site_ids}
        lookup = lookup.loc[lookup["site_id"].astype("int64").isin(site_id_set)].copy()

    lookup = lookup[
        ["site_id", "device_id", "circuit_id", "circuit_type", "circuit_polarity"]
    ].copy()
    lookup["site_id"] = lookup["site_id"].astype("int64")
    lookup["circuit_id"] = lookup["circuit_id"].astype("int64")
    lookup["circuit_polarity"] = pd.to_numeric(lookup["circuit_polarity"], errors="coerce").fillna(1.0)
    return lookup


def sites_with_required_trio(circuit_meta: pd.DataFrame) -> list[int]:
    """Return site IDs that have all required net-load, PV, and battery circuits."""

    type_sets = (
        circuit_meta.loc[circuit_meta["circuit_type"].isin(REQUIRED_CIRCUIT_TYPES)]
        .groupby("site_id")["circuit_type"]
        .agg(lambda values: set(values))
    )
    site_ids = [
        int(site_id)
        for site_id, circuit_types in type_sets.items()
        if set(REQUIRED_CIRCUIT_TYPES).issubset(circuit_types)
    ]
    return sorted(site_ids)


def read_daily_circuit_rows(parquet_path: Path, lookup: pd.DataFrame) -> pd.DataFrame:
    """Read one raw daily parquet file and attach site/circuit metadata."""

    daily = pd.read_parquet(parquet_path, columns=["circuit_id", "t_stamp", "power"])
    if daily.empty:
        return daily

    daily["circuit_id"] = daily["circuit_id"].astype("int64")
    daily = daily.merge(lookup, on="circuit_id", how="inner")
    if daily.empty:
        return daily

    daily["t_stamp"] = pd.to_datetime(daily["t_stamp"], errors="coerce").dt.tz_localize(None)
    daily["power"] = pd.to_numeric(daily["power"], errors="coerce")
    daily = daily.dropna(subset=["t_stamp", "power"])
    daily["raw_power"] = daily["power"]
    daily["adjusted_power"] = daily["power"] * daily["circuit_polarity"]
    return daily


def build_site_panel(rows: pd.DataFrame, value_column: str, circuit_types: Iterable[str]) -> pd.DataFrame:
    """Sum duplicate circuits/phases into one wide site-time panel."""

    if rows.empty:
        columns = ["site_id", "t_stamp", *list(circuit_types)]
        return pd.DataFrame(columns=columns)

    grouped = (
        rows.groupby(["site_id", "t_stamp", "circuit_type"], observed=True)[value_column]
        .sum()
        .reset_index()
    )
    wide = (
        grouped.pivot_table(
            index=["site_id", "t_stamp"],
            columns="circuit_type",
            values=value_column,
            aggfunc="sum",
        )
        .reset_index()
        .rename_axis(columns=None)
    )
    for circuit_type in circuit_types:
        if circuit_type not in wide.columns:
            wide[circuit_type] = np.nan
    return wide[["site_id", "t_stamp", *list(circuit_types)]]


def derive_profile_series(panel: pd.DataFrame, profile_id: str) -> pd.DataFrame:
    """Apply one candidate signal-definition profile to a site-time source panel."""

    required = ["site_id", "t_stamp", *REQUIRED_CIRCUIT_TYPES]
    if panel.empty:
        return pd.DataFrame(columns=["site_id", "t_stamp", *TARGET_COLUMNS, "pv_generation_kW", "battery_net_charge_kW", "battery_net_discharge_kW"])

    complete = panel.loc[panel[REQUIRED_CIRCUIT_TYPES].notna().all(axis=1), required].copy()
    if complete.empty:
        return pd.DataFrame(columns=["site_id", "t_stamp", *TARGET_COLUMNS, "pv_generation_kW", "battery_net_charge_kW", "battery_net_discharge_kW"])

    ac = complete["ac_load_net"] / 1000.0
    pv = complete["pv_site_net"] / 1000.0
    battery = complete["battery_storage"] / 1000.0

    result = complete[["site_id", "t_stamp"]].copy()

    if profile_id == CURRENT_PROFILE:
        result["pv_generation_kW"] = pv
        result["battery_net_charge_kW"] = battery
        result["battery_net_discharge_kW"] = -battery
        result["underlying_load_kW"] = ac + pv - battery
        result["net_load_with_pv_kW"] = result["underlying_load_kW"] - result["pv_generation_kW"]
        result["net_load_with_pv_and_battery_kW"] = ac
    elif profile_id == "battery_discharge_positive_adjusted":
        result["pv_generation_kW"] = pv
        result["battery_net_charge_kW"] = -battery
        result["battery_net_discharge_kW"] = battery
        result["underlying_load_kW"] = ac + pv + battery
        result["net_load_with_pv_kW"] = result["underlying_load_kW"] - result["pv_generation_kW"]
        result["net_load_with_pv_and_battery_kW"] = ac
    elif profile_id == "raw_processing_like":
        result["battery_net_discharge_kW"] = -battery
        result["battery_net_charge_kW"] = battery
        result["underlying_load_kW"] = ac + pv
        result["pv_generation_kW"] = pv - battery
        result["net_load_with_pv_kW"] = result["underlying_load_kW"] - result["pv_generation_kW"]
        result["net_load_with_pv_and_battery_kW"] = (
            result["underlying_load_kW"]
            - result["pv_generation_kW"]
            - result["battery_net_discharge_kW"]
        )
    else:
        raise ValueError(f"Unknown profile_id: {profile_id}")

    return result[
        [
            "site_id",
            "t_stamp",
            "underlying_load_kW",
            "pv_generation_kW",
            "battery_net_charge_kW",
            "battery_net_discharge_kW",
            "net_load_with_pv_kW",
            "net_load_with_pv_and_battery_kW",
        ]
    ]


def add_fixed_aest_datetime(frame: pd.DataFrame) -> pd.DataFrame:
    """Add fixed-AEST naive datetimes by applying UTC+10 to raw timestamps."""

    result = frame.copy()
    result["datetime"] = pd.to_datetime(result["t_stamp"]) + pd.Timedelta(hours=FIXED_AEST_OFFSET_HOURS)
    return result


def empty_metric_record() -> dict[str, float]:
    """Create a mutable metric accumulator for one formula profile."""

    return {
        "rows": 0,
        "underlying_negative_count": 0,
        "underlying_min_kW": math.inf,
        "pv_midday_count": 0,
        "pv_midday_positive_count": 0,
        "pv_night_count": 0,
        "pv_night_near_zero_count": 0,
        "battery_midday_count": 0,
        "battery_midday_charge_positive_count": 0,
        "battery_evening_count": 0,
        "battery_evening_discharge_positive_count": 0,
        "direct_ac_count": 0,
        "direct_ac_abs_error_sum": 0.0,
        "direct_pv_count": 0,
        "direct_pv_abs_error_sum": 0.0,
    }


def update_metric_record(record: dict[str, float], derived: pd.DataFrame, source_panel: pd.DataFrame) -> None:
    """Update physical-plausibility diagnostics from one derived daily chunk."""

    if derived.empty:
        return

    diagnostic = add_fixed_aest_datetime(derived)
    hour = diagnostic["datetime"].dt.hour
    rows = len(diagnostic)
    record["rows"] += rows

    underlying = diagnostic["underlying_load_kW"]
    record["underlying_negative_count"] += int(underlying.lt(NEGATIVE_LOAD_TOLERANCE_KW).sum())
    current_min = underlying.min(skipna=True)
    if pd.notna(current_min):
        record["underlying_min_kW"] = min(record["underlying_min_kW"], float(current_min))

    midday = hour.between(10, 14)
    night = hour.between(0, 4)
    evening = hour.between(18, 22)

    record["pv_midday_count"] += int(midday.sum())
    record["pv_midday_positive_count"] += int(diagnostic.loc[midday, "pv_generation_kW"].gt(0.05).sum())
    record["pv_night_count"] += int(night.sum())
    record["pv_night_near_zero_count"] += int(diagnostic.loc[night, "pv_generation_kW"].abs().le(0.05).sum())

    record["battery_midday_count"] += int(midday.sum())
    record["battery_midday_charge_positive_count"] += int(
        diagnostic.loc[midday, "battery_net_charge_kW"].gt(0.05).sum()
    )
    record["battery_evening_count"] += int(evening.sum())
    record["battery_evening_discharge_positive_count"] += int(
        diagnostic.loc[evening, "battery_net_discharge_kW"].gt(0.05).sum()
    )

    direct_columns = [column for column in DIRECT_CIRCUIT_TYPES if column in source_panel.columns]
    if not direct_columns:
        return

    direct_panel = source_panel[["site_id", "t_stamp", *direct_columns]].copy()
    comparison = diagnostic.merge(direct_panel, on=["site_id", "t_stamp"], how="left")

    if "ac_load" in comparison.columns:
        mask = comparison["ac_load"].notna() & comparison["underlying_load_kW"].notna()
        if mask.any():
            record["direct_ac_count"] += int(mask.sum())
            record["direct_ac_abs_error_sum"] += float(
                (comparison.loc[mask, "underlying_load_kW"] - comparison.loc[mask, "ac_load"] / 1000.0)
                .abs()
                .sum()
            )

    if "pv_site" in comparison.columns:
        mask = comparison["pv_site"].notna() & comparison["pv_generation_kW"].notna()
        if mask.any():
            record["direct_pv_count"] += int(mask.sum())
            record["direct_pv_abs_error_sum"] += float(
                (comparison.loc[mask, "pv_generation_kW"] - comparison.loc[mask, "pv_site"] / 1000.0)
                .abs()
                .sum()
            )


def daily_site_metrics(derived: pd.DataFrame) -> pd.DataFrame:
    """Aggregate complete source observations into per-site, per-AEST-day metrics."""

    if derived.empty:
        return pd.DataFrame(
            columns=["site_id", "date", "observed_count", "meaningful_negative_count", "min_underlying_load_kW"]
        )

    with_datetime = add_fixed_aest_datetime(derived)
    with_datetime["date"] = with_datetime["datetime"].dt.floor("D")
    with_datetime["meaningful_negative"] = with_datetime["underlying_load_kW"].lt(NEGATIVE_LOAD_TOLERANCE_KW)
    grouped = (
        with_datetime.groupby(["site_id", "date"], observed=True)
        .agg(
            observed_count=("underlying_load_kW", "size"),
            meaningful_negative_count=("meaningful_negative", "sum"),
            min_underlying_load_kW=("underlying_load_kW", "min"),
        )
        .reset_index()
    )
    grouped["site_id"] = grouped["site_id"].astype("int64")
    return grouped


def finalize_metric_records(metric_records: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Convert diagnostic accumulators into a scored profile summary table."""

    rows = []
    for profile_id, record in metric_records.items():
        rows_count = max(record["rows"], 1)

        def pct(numerator: float, denominator: float) -> float:
            if denominator <= 0:
                return np.nan
            return 100.0 * numerator / denominator

        underlying_positive_pct = 100.0 - pct(record["underlying_negative_count"], rows_count)
        pv_midday_positive_pct = pct(record["pv_midday_positive_count"], record["pv_midday_count"])
        pv_night_near_zero_pct = pct(record["pv_night_near_zero_count"], record["pv_night_count"])
        battery_midday_charge_pct = pct(
            record["battery_midday_charge_positive_count"], record["battery_midday_count"]
        )
        battery_evening_discharge_pct = pct(
            record["battery_evening_discharge_positive_count"], record["battery_evening_count"]
        )

        score_components = [
            0.30 * underlying_positive_pct,
            0.20 * pv_midday_positive_pct,
            0.20 * pv_night_near_zero_pct,
            0.15 * battery_midday_charge_pct,
            0.15 * battery_evening_discharge_pct,
        ]
        diagnostic_score = float(np.nansum(score_components))

        direct_ac_mae = (
            record["direct_ac_abs_error_sum"] / record["direct_ac_count"]
            if record["direct_ac_count"] > 0
            else np.nan
        )
        direct_pv_mae = (
            record["direct_pv_abs_error_sum"] / record["direct_pv_count"]
            if record["direct_pv_count"] > 0
            else np.nan
        )

        rows.append(
            {
                "profile_id": profile_id,
                "label": FORMULA_PROFILES[profile_id].label,
                "basis": FORMULA_PROFILES[profile_id].basis,
                "battery_positive_meaning": FORMULA_PROFILES[profile_id].battery_positive_meaning,
                "rows": int(record["rows"]),
                "diagnostic_score": round(diagnostic_score, 3),
                "underlying_positive_pct": round(underlying_positive_pct, 3),
                "underlying_negative_count": int(record["underlying_negative_count"]),
                "underlying_min_kW": round(record["underlying_min_kW"], 3),
                "pv_midday_positive_pct": round(pv_midday_positive_pct, 3),
                "pv_night_near_zero_pct": round(pv_night_near_zero_pct, 3),
                "battery_midday_charge_positive_pct": round(battery_midday_charge_pct, 3),
                "battery_evening_discharge_positive_pct": round(battery_evening_discharge_pct, 3),
                "direct_ac_count": int(record["direct_ac_count"]),
                "direct_ac_mae_kW": round(direct_ac_mae, 3) if pd.notna(direct_ac_mae) else np.nan,
                "direct_pv_count": int(record["direct_pv_count"]),
                "direct_pv_mae_kW": round(direct_pv_mae, 3) if pd.notna(direct_pv_mae) else np.nan,
            }
        )

    return pd.DataFrame(rows).sort_values("diagnostic_score", ascending=False).reset_index(drop=True)


def choose_profile(metric_summary: pd.DataFrame) -> tuple[str, str]:
    """Choose the signal definition using the evidence-gated current-hypothesis rule."""

    current_row = metric_summary.loc[metric_summary["profile_id"].eq(CURRENT_PROFILE)].iloc[0]
    best_row = metric_summary.iloc[0]
    score_gap = float(best_row["diagnostic_score"] - current_row["diagnostic_score"])

    if best_row["profile_id"] != CURRENT_PROFILE and score_gap >= 10.0:
        reason = (
            f"Selected {best_row['profile_id']} because its diagnostic score exceeds the current hypothesis "
            f"by {score_gap:.2f} points."
        )
        return str(best_row["profile_id"]), reason

    reason = (
        "Selected the documented polarity-adjusted hypothesis because no alternative cleared the "
        "10-point evidence gate over the current hypothesis."
    )
    return CURRENT_PROFILE, reason


def combine_daily_metrics(parts: list[pd.DataFrame]) -> pd.DataFrame:
    """Combine daily metric chunks and collapse dates split across UTC source files."""

    if not parts:
        return pd.DataFrame(
            columns=["site_id", "date", "observed_count", "meaningful_negative_count", "min_underlying_load_kW"]
        )

    combined = pd.concat(parts, ignore_index=True)
    combined = (
        combined.groupby(["site_id", "date"], observed=True)
        .agg(
            observed_count=("observed_count", "sum"),
            meaningful_negative_count=("meaningful_negative_count", "sum"),
            min_underlying_load_kW=("min_underlying_load_kW", "min"),
        )
        .reset_index()
    )
    combined["site_id"] = combined["site_id"].astype("int64")
    return combined


def choose_window_and_households(
    daily_metrics: pd.DataFrame,
    site_meta: pd.DataFrame,
    selected_profile_id: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Choose a 365-day window and deterministic 100-household sample."""

    if daily_metrics.empty:
        raise ValueError("No daily metrics were available for household selection.")

    all_dates = pd.date_range(
        daily_metrics["date"].min().normalize(),
        daily_metrics["date"].max().normalize(),
        freq="D",
    )
    site_ids = sorted(daily_metrics["site_id"].unique())

    observed = (
        daily_metrics.pivot(index="date", columns="site_id", values="observed_count")
        .reindex(index=all_dates, columns=site_ids)
        .fillna(0)
    )
    negatives = (
        daily_metrics.pivot(index="date", columns="site_id", values="meaningful_negative_count")
        .reindex(index=all_dates, columns=site_ids)
        .fillna(0)
    )
    min_load = (
        daily_metrics.pivot(index="date", columns="site_id", values="min_underlying_load_kW")
        .reindex(index=all_dates, columns=site_ids)
    )

    observed_roll = observed.rolling(N_DAYS, min_periods=N_DAYS).sum()
    negatives_roll = negatives.rolling(N_DAYS, min_periods=N_DAYS).sum()
    min_load_roll = min_load.rolling(N_DAYS, min_periods=1).min()
    first_observed_date = observed.gt(0).idxmax()
    first_observed_date = first_observed_date.where(observed.gt(0).any(axis=0), pd.NaT)

    best_selection: pd.DataFrame | None = None
    best_info: dict[str, object] | None = None
    best_score: tuple[float, ...] | None = None

    for missingness_threshold_pct in SELECTION_MISSINGNESS_THRESHOLDS_PCT:
        for end_date in all_dates[N_DAYS - 1 :]:
            start_date = end_date - pd.Timedelta(days=N_DAYS - 1)
            obs_row = observed_roll.loc[end_date]
            neg_row = negatives_roll.loc[end_date]
            min_row = min_load_roll.loc[end_date]

            candidate_pool = pd.DataFrame(
                {
                    "site_id": site_ids,
                    "selected_overlap_observed_timestamps_pre_fill": obs_row.to_numpy(dtype=float),
                    "selected_overlap_meaningful_negative_count_pre_fill": neg_row.to_numpy(dtype=float),
                    "selected_overlap_min_underlying_load_kW_pre_fill": min_row.to_numpy(dtype=float),
                }
            )
            candidate_pool["selected_overlap_missingness_pct_pre_fill"] = (
                100.0
                * (EXPECTED_5MIN_ROWS - candidate_pool["selected_overlap_observed_timestamps_pre_fill"])
                / EXPECTED_5MIN_ROWS
            )
            latest_allowed_first_observed = start_date - pd.Timedelta(days=SELECTION_PRIOR_FILL_BUFFER_DAYS)
            candidate_pool["first_observed_date"] = candidate_pool["site_id"].map(first_observed_date)
            candidate_pool = candidate_pool.loc[
                candidate_pool["selected_overlap_missingness_pct_pre_fill"].le(missingness_threshold_pct)
                & candidate_pool["first_observed_date"].le(latest_allowed_first_observed)
            ].copy()
            if candidate_pool.shape[0] < N_HOUSEHOLDS:
                continue

            candidate_pool["has_meaningful_negative_underlying_load_pre_fill"] = candidate_pool[
                "selected_overlap_meaningful_negative_count_pre_fill"
            ].gt(0)
            candidate = candidate_pool.sort_values(
                [
                    "has_meaningful_negative_underlying_load_pre_fill",
                    "selected_overlap_missingness_pct_pre_fill",
                    "selected_overlap_meaningful_negative_count_pre_fill",
                    "site_id",
                ],
                kind="mergesort",
            ).head(N_HOUSEHOLDS)

            score = (
                float(missingness_threshold_pct),
                float(candidate["has_meaningful_negative_underlying_load_pre_fill"].sum()),
                float(candidate["selected_overlap_missingness_pct_pre_fill"].mean()),
                float(candidate["selected_overlap_meaningful_negative_count_pre_fill"].sum()),
                float(candidate["selected_overlap_missingness_pct_pre_fill"].max()),
                float(start_date.toordinal()),
            )
            if best_score is None or score < best_score:
                best_score = score
                best_selection = candidate.copy()
                best_info = {
                    "selected_profile_id": selected_profile_id,
                    "selected_overlap_start": start_date,
                    "selected_overlap_end": end_date + pd.Timedelta(days=1) - pd.Timedelta(minutes=5),
                    "selected_overlap_days": N_DAYS,
                    "selected_overlap_expected_timestamps": EXPECTED_5MIN_ROWS,
                    "negative_load_tolerance_kW": NEGATIVE_LOAD_TOLERANCE_KW,
                    "selection_missingness_threshold_pct": missingness_threshold_pct,
                    "selection_prior_fill_buffer_days": SELECTION_PRIOR_FILL_BUFFER_DAYS,
                    "selection_candidate_pool_size": int(candidate_pool.shape[0]),
                    "selected_sites_with_meaningful_negative_pre_fill": int(score[1]),
                    "selected_total_meaningful_negative_pre_fill": int(score[3]),
                    "selected_mean_missingness_pct_pre_fill": score[2],
                    "selected_max_missingness_pct_pre_fill": score[4],
                }

    if best_selection is None or best_info is None:
        raise ValueError("Could not find a 365-day window supporting 100 candidate households.")

    selected = best_selection.reset_index(drop=True)
    selected["selection_rank"] = np.arange(1, selected.shape[0] + 1)
    selected["is_selected_sample"] = 1

    selected = selected.merge(
        site_meta[
            [
                "site_id",
                "state",
                "postcode",
                "latitude",
                "longitude",
                "dc_capacity_kw",
                "ac_capacity_kw",
                "monitoring_start",
            ]
        ],
        on="site_id",
        how="left",
    )
    return selected, best_info


def write_signal_report(
    metric_summary: pd.DataFrame,
    selected_profile_id: str,
    selection_reason: str,
    selection_info: dict[str, object],
) -> None:
    """Write the Markdown diagnostic report used to justify the chosen signal definition."""

    report_lines = [
        "# SA BESS Signal Diagnostics Report",
        "",
        "## Executive Summary",
        "",
        f"- Selected signal definition: `{selected_profile_id}`.",
        f"- Selection rule: {selection_reason}",
        f"- Fixed time basis for outputs: AEST = UTC+{FIXED_AEST_OFFSET_HOURS}, stored as naive datetimes.",
        f"- Meaningful negative underlying-load threshold: `{NEGATIVE_LOAD_TOLERANCE_KW} kW` per household.",
        "",
        "## Candidate Formula Profiles",
        "",
    ]

    for profile in FORMULA_PROFILES.values():
        report_lines.extend(
            [
                f"### `{profile.profile_id}`",
                "",
                f"- Label: {profile.label}",
                f"- Power basis: `{profile.basis}`",
                f"- Positive `battery_storage` interpretation: {profile.battery_positive_meaning}",
                f"- Notes: {profile.notes}",
                "",
            ]
        )

    report_lines.extend(
        [
            "## Diagnostic Score Table",
            "",
            dataframe_to_markdown(metric_summary),
            "",
            "The score is a weighted physical-plausibility screen: underlying-load positivity, daylight PV positivity,",
            "overnight PV near-zero behaviour, midday battery charge behaviour, and evening battery discharge behaviour.",
            "It is not treated as proof of the raw data definition; it is a structured confidence check.",
            "",
            "## Selected 100-Household Window",
            "",
            f"- Start: `{selection_info['selected_overlap_start']}` fixed AEST.",
            f"- End: `{selection_info['selected_overlap_end']}` fixed AEST.",
            f"- Expected 5-minute timestamps per site: `{selection_info['selected_overlap_expected_timestamps']}`.",
            f"- Selection missingness threshold: `{selection_info['selection_missingness_threshold_pct']}%`.",
            f"- Prior same-clock fill buffer: `{selection_info['selection_prior_fill_buffer_days']}` days.",
            f"- Candidate pool size at threshold: `{selection_info['selection_candidate_pool_size']}`.",
            f"- Sites with meaningful negative underlying load before fill: `{selection_info['selected_sites_with_meaningful_negative_pre_fill']}`.",
            f"- Mean missingness before fill: `{selection_info['selected_mean_missingness_pct_pre_fill']:.4f}%`.",
            f"- Max missingness before fill: `{selection_info['selected_max_missingness_pct_pre_fill']:.4f}%`.",
            "",
            "## Interpretation",
            "",
            "The workflow keeps the documented polarity-adjusted hypothesis unless an alternative beats it by at least",
            "10 diagnostic-score points. This protects against switching definitions on weak or noisy evidence while still",
            "allowing a clear contradiction to override the default.",
            "",
        ]
    )

    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without relying on optional tabulate."""

    text_frame = frame.copy()
    for column in text_frame.columns:
        text_frame[column] = text_frame[column].map(lambda value: "" if pd.isna(value) else str(value))

    headers = list(text_frame.columns)
    rows = text_frame.values.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def run_diagnostics_and_select() -> tuple[str, pd.DataFrame, dict[str, object], pd.DataFrame]:
    """Run full-source diagnostics and select the 100-household analysis sample."""

    ensure_directories()
    site_meta, circuit_meta = load_metadata()
    trio_site_ids = sites_with_required_trio(circuit_meta)
    lookup = build_circuit_lookup(circuit_meta, ALL_DIAGNOSTIC_CIRCUIT_TYPES, trio_site_ids)
    paths = parquet_paths()

    metric_records = {profile_id: empty_metric_record() for profile_id in FORMULA_PROFILES}
    daily_parts = {profile_id: [] for profile_id in FORMULA_PROFILES}

    for idx, parquet_path in enumerate(paths, start=1):
        rows = read_daily_circuit_rows(parquet_path, lookup)
        if rows.empty:
            continue

        raw_panel = build_site_panel(rows, "raw_power", ALL_DIAGNOSTIC_CIRCUIT_TYPES)
        adjusted_panel = build_site_panel(rows, "adjusted_power", ALL_DIAGNOSTIC_CIRCUIT_TYPES)

        for profile_id, profile in FORMULA_PROFILES.items():
            source_panel = adjusted_panel if profile.basis == "adjusted_power" else raw_panel
            derived = derive_profile_series(source_panel, profile_id)
            update_metric_record(metric_records[profile_id], derived, source_panel)
            daily_parts[profile_id].append(daily_site_metrics(derived))

        if idx == 1 or idx % 25 == 0 or idx == len(paths):
            print(f"[diagnostics] processed {idx:,}/{len(paths):,} parquet files")

    metric_summary = finalize_metric_records(metric_records)
    selected_profile_id, selection_reason = choose_profile(metric_summary)
    selected_daily_metrics = combine_daily_metrics(daily_parts[selected_profile_id])
    selected_households, selection_info = choose_window_and_households(
        selected_daily_metrics,
        site_meta,
        selected_profile_id,
    )

    metric_summary.to_csv(DIAGNOSTICS_DIR / "sa_bess_signal_diagnostic_scores.csv", index=False)
    selected_daily_metrics.to_csv(INTERMEDIATE_DIR / "sa_bess_daily_site_metrics.csv", index=False)
    selected_households.to_csv(INTERMEDIATE_DIR / "sa_bess_selected_households_pre_fill.csv", index=False)
    write_signal_report(metric_summary, selected_profile_id, selection_reason, selection_info)

    print(f"[diagnostics] selected profile: {selected_profile_id}")
    print(f"[diagnostics] report written: {REPORT_PATH}")
    return selected_profile_id, selected_households, selection_info, metric_summary


def load_selected_source_history(
    selected_site_ids: list[int],
    selected_profile_id: str,
    window_end: pd.Timestamp,
) -> pd.DataFrame:
    """Load and derive selected-site source history up to the end of the target window."""

    _, circuit_meta = load_metadata()
    profile = FORMULA_PROFILES[selected_profile_id]
    lookup = build_circuit_lookup(circuit_meta, REQUIRED_CIRCUIT_TYPES, selected_site_ids)

    source_parts = []
    for idx, parquet_path in enumerate(parquet_paths(), start=1):
        rows = read_daily_circuit_rows(parquet_path, lookup)
        if rows.empty:
            continue

        source_panel = build_site_panel(rows, profile.basis, REQUIRED_CIRCUIT_TYPES)
        derived = derive_profile_series(source_panel, selected_profile_id)
        if derived.empty:
            continue

        derived = add_fixed_aest_datetime(derived)
        derived = derived.loc[derived["datetime"].le(window_end)].copy()
        if not derived.empty:
            source_parts.append(derived[["site_id", "datetime", *TARGET_COLUMNS]])

        if idx == 1 or idx % 25 == 0 or idx == len(parquet_paths()):
            print(f"[selected-source] processed {idx:,}/{len(parquet_paths()):,} parquet files")

    if not source_parts:
        raise ValueError("No selected-source rows were loaded.")

    source = pd.concat(source_parts, ignore_index=True)
    source["site_id"] = source["site_id"].astype("int64")
    source["datetime"] = pd.to_datetime(source["datetime"])
    source = (
        source.groupby(["site_id", "datetime"], observed=True)[TARGET_COLUMNS]
        .sum()
        .reset_index()
        .sort_values(["site_id", "datetime"])
    )
    return source


def fill_site_from_previous_day_same_clock(
    site_source: pd.DataFrame,
    site_id: int,
    overlap_index: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Fill one selected site using prior observations from the same clock time."""

    source = (
        site_source.loc[site_source["site_id"].eq(site_id), ["datetime", *TARGET_COLUMNS]]
        .drop_duplicates(subset=["datetime"], keep="last")
        .sort_values("datetime")
    )
    values = source.set_index("datetime")[TARGET_COLUMNS]
    pre_fill = values.reindex(overlap_index)
    pre_observed = int(pre_fill.notna().all(axis=1).sum())

    combined_index = values.index.union(overlap_index).sort_values()
    combined = values.reindex(combined_index)
    combined["clock_time"] = combined.index.strftime("%H:%M:%S")
    combined[TARGET_COLUMNS] = combined.groupby("clock_time", sort=False)[TARGET_COLUMNS].ffill()
    filled = combined.loc[overlap_index, TARGET_COLUMNS].reset_index().rename(columns={"index": "datetime"})
    filled.insert(0, "site_id", int(site_id))

    post_observed = int(filled[TARGET_COLUMNS].notna().all(axis=1).sum())
    negative_pre = pre_fill["underlying_load_kW"].lt(NEGATIVE_LOAD_TOLERANCE_KW)
    negative_post = filled["underlying_load_kW"].lt(NEGATIVE_LOAD_TOLERANCE_KW)

    metrics = {
        "site_id": int(site_id),
        "selected_overlap_observed_timestamps_pre_fill": pre_observed,
        "selected_overlap_missingness_pct_pre_fill": 100.0 * (len(overlap_index) - pre_observed) / len(overlap_index),
        "selected_overlap_observed_timestamps_post_fill": post_observed,
        "selected_overlap_missingness_pct_post_fill": 100.0 * (len(overlap_index) - post_observed) / len(overlap_index),
        "meaningful_negative_underlying_load_count_pre_fill": int(negative_pre.sum()),
        "meaningful_negative_underlying_load_pct_pre_fill": 100.0 * int(negative_pre.sum()) / len(overlap_index),
        "meaningful_negative_underlying_load_count_post_fill": int(negative_post.sum()),
        "meaningful_negative_underlying_load_pct_post_fill": 100.0 * int(negative_post.sum()) / len(overlap_index),
        "min_underlying_load_kW_pre_fill": float(pre_fill["underlying_load_kW"].min(skipna=True)),
        "min_underlying_load_kW_post_fill": float(filled["underlying_load_kW"].min(skipna=True)),
    }
    return filled, metrics


def write_pynnlf_datasets(aggregate_5min: pd.DataFrame) -> pd.DataFrame:
    """Write 30-minute mean-power PyNNLF datasets in the publication workspace and root data folder."""

    aggregate_30min = (
        aggregate_5min.set_index("datetime")[TARGET_COLUMNS]
        .resample("30min", label="left", closed="left")
        .mean()
        .reset_index()
    )
    aggregate_30min.to_csv(PROCESSED_DIR / "sa_bess_aggregate_30min_mean_power.csv", index=False)

    dataset_rows = []
    for dataset_id, spec in DATASET_SPECS.items():
        dataset = aggregate_30min[["datetime", spec["source_column"]]].rename(
            columns={spec["source_column"]: "netload_kW"}
        )
        if dataset["netload_kW"].isna().any():
            raise ValueError(f"{dataset_id} contains missing netload_kW values after 30-minute mean aggregation.")
        if dataset.shape[0] != 17_520:
            raise ValueError(f"{dataset_id} expected 17,520 rows, found {dataset.shape[0]:,}.")

        workspace_path = WORKSPACE_DIR / "data" / spec["filename"]
        dataset.to_csv(workspace_path, index=False)

        dataset_rows.append(
            {
                "dataset_id": dataset_id,
                "filename": spec["filename"],
                "label": spec["label"],
                "rows": dataset.shape[0],
                "start": dataset["datetime"].min(),
                "end": dataset["datetime"].max(),
                "workspace_path": str(workspace_path),
            }
        )

    dataset_summary = pd.DataFrame(dataset_rows)
    dataset_summary.to_csv(PROCESSED_DIR / "sa_bess_pynnlf_dataset_summary.csv", index=False)
    return dataset_summary


def process_selected_sample(
    selected_profile_id: str,
    selected_households: pd.DataFrame,
    selection_info: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Derive, fill, aggregate, and export the selected 100-household BESS sample."""

    window_start = pd.Timestamp(selection_info["selected_overlap_start"])
    window_end = pd.Timestamp(selection_info["selected_overlap_end"])
    overlap_index = pd.date_range(window_start, window_end, freq="5min", name="datetime")
    if len(overlap_index) != EXPECTED_5MIN_ROWS:
        raise ValueError(f"Expected {EXPECTED_5MIN_ROWS:,} timestamps, found {len(overlap_index):,}.")

    selected_site_ids = selected_households["site_id"].astype("int64").tolist()
    selected_source = load_selected_source_history(selected_site_ids, selected_profile_id, window_end)
    selected_source.to_parquet(INTERMEDIATE_DIR / "sa_bess_selected_source_history.parquet", index=False)

    filled_parts = []
    fill_metric_rows = []
    for rank, site_id in enumerate(selected_site_ids, start=1):
        filled, metrics = fill_site_from_previous_day_same_clock(selected_source, site_id, overlap_index)
        filled_parts.append(filled)
        metrics["selection_rank"] = rank
        fill_metric_rows.append(metrics)
        if rank == 1 or rank % 10 == 0 or rank == len(selected_site_ids):
            print(f"[fill] processed {rank:,}/{len(selected_site_ids):,} selected households")

    site_timeseries = pd.concat(filled_parts, ignore_index=True)
    fill_metrics = pd.DataFrame(fill_metric_rows)
    if site_timeseries[TARGET_COLUMNS].isna().any().any():
        missing_cells = int(site_timeseries[TARGET_COLUMNS].isna().sum().sum())
        raise ValueError(f"Post-fill selected site timeseries still contains {missing_cells:,} missing cells.")

    summary = selected_households.drop(
        columns=[
            "selected_overlap_observed_timestamps_pre_fill",
            "selected_overlap_missingness_pct_pre_fill",
            "selected_overlap_meaningful_negative_count_pre_fill",
            "selected_overlap_min_underlying_load_kW_pre_fill",
            "has_meaningful_negative_underlying_load_pre_fill",
        ],
        errors="ignore",
    ).merge(fill_metrics, on=["site_id", "selection_rank"], how="left")
    summary.insert(1, "formula_profile", selected_profile_id)
    summary.insert(2, "selected_overlap_start", window_start)
    summary.insert(3, "selected_overlap_end", window_end)
    summary.insert(4, "selected_overlap_days", N_DAYS)
    summary.insert(5, "selected_overlap_expected_timestamps", EXPECTED_5MIN_ROWS)
    summary.insert(6, "negative_load_tolerance_kW", NEGATIVE_LOAD_TOLERANCE_KW)

    site_timeseries.to_parquet(INTERMEDIATE_DIR / "sa_bess_selected_site_timeseries_5min.parquet", index=False)
    summary.to_csv(PROCESSED_DIR / "sa_bess_selected_household_summary.csv", index=False)

    aggregate_5min = (
        site_timeseries.groupby("datetime", as_index=False, observed=True)[TARGET_COLUMNS]
        .sum()
        .sort_values("datetime")
        .reset_index(drop=True)
    )
    if aggregate_5min.shape[0] != EXPECTED_5MIN_ROWS:
        raise ValueError(f"Aggregate expected {EXPECTED_5MIN_ROWS:,} rows, found {aggregate_5min.shape[0]:,}.")

    aggregate_5min.to_csv(PROCESSED_DIR / "sa_bess_aggregate_5min.csv", index=False)
    aggregate_5min.to_parquet(PROCESSED_DIR / "sa_bess_aggregate_5min.parquet", index=False)
    write_pynnlf_datasets(aggregate_5min)

    return aggregate_5min, site_timeseries, summary


def write_experiment_specs() -> None:
    """Write the smoke and full SA BESS PyNNLF batch specs."""

    specs_dir = WORKSPACE_DIR / "specs"
    smoke = {
        "datasets": ["ds16"],
        "forecast_horizons": ["fh1"],
        "model_and_hp": [["m1", "hp1"]],
    }
    full = {
        "datasets": ["ds16", "ds17", "ds18"],
        "forecast_horizons": ["fh1"],
        "model_and_hp": MODEL_AND_HP,
    }
    (specs_dir / "sa_bess_smoke.yaml").write_text(yaml.safe_dump(smoke, sort_keys=False), encoding="utf-8")
    (specs_dir / "sa_bess_batch.yaml").write_text(yaml.safe_dump(full, sort_keys=False), encoding="utf-8")


def notebook(cells: list[tuple[str, str]]) -> nbf.NotebookNode:
    """Create a version-4 notebook from pairs of markdown/code cell definitions."""

    nb = nbf.v4.new_notebook()
    nb.cells = []
    for cell_type, source in cells:
        if cell_type == "markdown":
            nb.cells.append(nbf.v4.new_markdown_cell(source))
        elif cell_type == "code":
            nb.cells.append(nbf.v4.new_code_cell(source))
        else:
            raise ValueError(f"Unknown cell_type: {cell_type}")
    return nb


def pynnlf_visualisation_cells() -> list[tuple[str, str]]:
    """Return cells for the PyNNLF SA BESS result visualisation notebook."""

    return [
        (
            "markdown",
            "# 6. Visualise PyNNLF SA BESS Results\n\n"
            "This notebook compares the three SA BESS datasets used in the PyNNLF experiment. "
            "The focus is the change in load shape, distribution, and model performance between "
            "underlying load, net load with PV, and net load with PV plus battery.\n\n"
            "Dataset provenance:\n"
            "- Source: Solar Analytics CICCADA BESS aggregate processed by the publication workflow.\n"
            "- Sample: 100 households.\n"
            "- Period: 2024-02-26 00:00 to 2025-02-24 23:30 fixed AEST.\n"
            "- Length: 17,520 half-hour rows per dataset, covering 365 days.\n"
            "- Signal profile: `current_polarity_adjusted`.\n"
            "- No missing values are expected in the exported 30-minute datasets.",
        ),
        (
            "markdown",
            "## Setup\n\nLoad the exported SA BESS datasets and PyNNLF result tables.",
        ),
        (
            "code",
            """from pathlib import Path
import sys

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

PROJECT_DIR = Path.cwd().parents[0] if Path.cwd().name == 'notebooks' else Path.cwd()
DATA_DIR = PROJECT_DIR / 'data'
RESULTS_DIR = PROJECT_DIR / 'results'
FIGURES_DIR = RESULTS_DIR / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

N_HOUSEHOLDS = 100
N_DAYS = 365
SIGNAL_PROFILE = 'current_polarity_adjusted'
DATASET_ORDER = ['underlying_load', 'net_load_with_pv', 'net_load_with_pv_battery']
DATASET_SPECS = {
    'underlying_load': {
        'filename': 'ds16_sa_bess_underlying_load_30min.csv',
        'label': 'Underlying load',
        'definition': 'Aggregate underlying household demand before PV and battery effects.',
    },
    'net_load_with_pv': {
        'filename': 'ds17_sa_bess_net_load_with_pv_30min.csv',
        'label': 'Net load with PV',
        'definition': 'Aggregate load after PV generation, before battery operation.',
    },
    'net_load_with_pv_battery': {
        'filename': 'ds18_sa_bess_net_load_with_pv_battery_30min.csv',
        'label': 'Net load with PV and battery',
        'definition': 'Aggregate load after PV generation and battery operation.',
    },
}
DATASET_LABELS = {key: spec['label'] for key, spec in DATASET_SPECS.items()}
COLORS = {
    'underlying_load': '#2f5597',
    'net_load_with_pv': '#70ad47',
    'net_load_with_pv_battery': '#c55a11',
}
BEST_BAR_HATCH = '///'

try:
    display
except NameError:
    def display(obj):
        print(obj)

plt.rcParams.update({
    'figure.dpi': 120,
    'savefig.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.25,
})


def save_figure(fig, filename):
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f'Saved: {path}')


series_frames = []
metadata_rows = []
expected_datetime = None

for dataset_key, spec in DATASET_SPECS.items():
    path = DATA_DIR / spec['filename']
    df = pd.read_csv(path, parse_dates=['datetime']).sort_values('datetime').reset_index(drop=True)
    expected_columns = {'datetime', 'netload_kW'}
    missing_columns = expected_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f'{path.name} is missing columns: {sorted(missing_columns)}')

    if expected_datetime is None:
        expected_datetime = df['datetime']
    elif not df['datetime'].equals(expected_datetime):
        raise ValueError(f'{path.name} does not share the same datetime index as the first dataset.')

    missing_count = int(df[['datetime', 'netload_kW']].isna().sum().sum())
    metadata_rows.append({
        'dataset': spec['label'],
        'file': spec['filename'],
        'rows': len(df),
        'start': df['datetime'].min(),
        'end': df['datetime'].max(),
        'missing_cells': missing_count,
        'households': N_HOUSEHOLDS,
        'days': N_DAYS,
        'signal_profile': SIGNAL_PROFILE,
        'definition': spec['definition'],
    })
    series_frames.append(df[['datetime', 'netload_kW']].rename(columns={'netload_kW': dataset_key}))

datasets = series_frames[0]
for frame in series_frames[1:]:
    datasets = datasets.merge(frame, on='datetime', how='inner', validate='one_to_one')

if datasets[DATASET_ORDER].isna().any().any():
    raise ValueError('The combined dataset table contains missing netload values.')

metadata = pd.DataFrame(metadata_rows)
nrmse = pd.read_csv(RESULTS_DIR / 'sa_bess_nrmse_comparison.csv')
nrmse_stddev = pd.read_csv(RESULTS_DIR / 'sa_bess_nrmse_stddev_comparison.csv')
expected_result_columns = ['model_hp', *DATASET_ORDER]

if list(nrmse.columns) != expected_result_columns:
    raise ValueError(f'nRMSE columns differ from expected columns: {list(nrmse.columns)}')
if list(nrmse_stddev.columns) != expected_result_columns:
    raise ValueError(f'nRMSE stddev columns differ from expected columns: {list(nrmse_stddev.columns)}')
if not nrmse['model_hp'].equals(nrmse_stddev['model_hp']):
    raise ValueError('nRMSE and nRMSE stddev tables do not have matching model_hp rows.')

nrmse = nrmse.set_index('model_hp')[DATASET_ORDER]
nrmse_stddev = nrmse_stddev.set_index('model_hp')[DATASET_ORDER]
metadata""",
        ),
        (
            "markdown",
            "## Dataset Description\n\nConfirm the provenance, size, date range, and signal definition for each exported dataset.",
        ),
        (
            "code",
            """display(metadata)
print(f'Combined dataset shape: {datasets.shape[0]:,} timestamps x {len(DATASET_ORDER)} series')
print(f'Datetime coverage: {datasets["datetime"].min()} to {datasets["datetime"].max()}')
print(f'Expected half-hour rows per dataset: {48 * N_DAYS:,}')""",
        ),
        (
            "markdown",
            "## Typical Weekly Profile\n\nAverage each half-hour slot of the week across the full year to compare load-shape changes.",
        ),
        (
            "code",
            """profile_source = datasets.copy()
profile_source['week_slot'] = (
    profile_source['datetime'].dt.dayofweek * 48
    + profile_source['datetime'].dt.hour * 2
    + profile_source['datetime'].dt.minute // 30
)
typical_week = profile_source.groupby('week_slot')[DATASET_ORDER].mean().reindex(range(7 * 48))

fig, ax = plt.subplots(figsize=(13, 5))
for dataset_key in DATASET_ORDER:
    ax.plot(
        typical_week.index,
        typical_week[dataset_key],
        label=DATASET_LABELS[dataset_key],
        color=COLORS[dataset_key],
        linewidth=2.0,
    )

for day in range(8):
    ax.axvline(day * 48, color='0.85', linewidth=0.8, zorder=0)

ax.set_xticks([day * 48 + 24 for day in range(7)])
ax.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
ax.set_ylabel('Aggregate load (kW)')
ax.set_title('Typical weekly profile of SA BESS aggregate datasets')
ax.legend(loc='upper left')
fig.tight_layout()
save_figure(fig, 'fig2_typical_week_profile.png')
plt.show()""",
        ),
        (
            "markdown",
            "## Distribution And Summary Statistics\n\nCompare the annual distribution of aggregate half-hourly load across the three dataset definitions.",
        ),
        (
            "code",
            """base_stats = datasets[DATASET_ORDER].agg(['mean', 'std', 'min', 'median', 'max']).T
quantile_stats = datasets[DATASET_ORDER].quantile([0.05, 0.25, 0.75, 0.95]).T
quantile_stats.columns = ['p05', 'p25', 'p75', 'p95']
summary_stats = pd.concat([base_stats[['mean', 'std', 'min']], quantile_stats[['p05', 'p25']], base_stats[['median']], quantile_stats[['p75', 'p95']], base_stats[['max']]], axis=1)
summary_stats = summary_stats.rename(index=DATASET_LABELS).round(3)
display(summary_stats)

fig, ax = plt.subplots(figsize=(8, 5))
datasets[DATASET_ORDER].rename(columns=DATASET_LABELS).plot.box(ax=ax)
ax.set_ylabel('Aggregate load (kW)')
ax.set_title('Annual half-hourly load distributions')
ax.tick_params(axis='x', rotation=0)
fig.tight_layout()
save_figure(fig, 'fig1_dataset_distributions.png')
plt.show()""",
        ),
        (
            "markdown",
            "## Dataset-Level Deltas\n\nShow how PV and PV plus battery move the typical weekly profile away from underlying load.",
        ),
        (
            "code",
            """typical_week_delta = pd.DataFrame({
    'PV effect: net load with PV minus underlying load': typical_week['net_load_with_pv'] - typical_week['underlying_load'],
    'PV and battery effect: net load with PV and battery minus underlying load': typical_week['net_load_with_pv_battery'] - typical_week['underlying_load'],
})

fig, ax = plt.subplots(figsize=(13, 5))
typical_week_delta.plot(ax=ax, linewidth=2.0)
for day in range(8):
    ax.axvline(day * 48, color='0.85', linewidth=0.8, zorder=0)
ax.axhline(0, color='0.25', linewidth=0.9)
ax.set_xticks([day * 48 + 24 for day in range(7)])
ax.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
ax.set_ylabel('Difference from underlying load (kW)')
ax.set_title('Typical weekly dataset deltas relative to underlying load')
ax.legend(loc='lower left')
fig.tight_layout()
save_figure(fig, 'fig3_typical_week_delta.png')
plt.show()""",
        ),
        (
            "markdown",
            "## Model Performance By Dataset\n\nCompare nRMSE by dataset definition. The model order follows the exported result table.",
        ),
        (
            "code",
            """display(nrmse.rename(columns=DATASET_LABELS).round(3))
display(nrmse_stddev.rename(columns=DATASET_LABELS).round(3))

plot_nrmse = nrmse.T.rename(index=DATASET_LABELS)
plot_stddev = nrmse_stddev.T
plot_stddev.index = plot_nrmse.index


def highlight_dataset_best(ax, value_table):
    bar_containers = [
        container
        for container in ax.containers
        if hasattr(container, 'patches') and len(container.patches) == len(value_table.index)
    ]
    column_to_container = dict(zip(value_table.columns, bar_containers))
    summary_rows = []

    for row_idx, dataset_label in enumerate(value_table.index):
        best_model = value_table.loc[dataset_label].idxmin()
        best_value = float(value_table.loc[dataset_label, best_model])
        patch = column_to_container[best_model].patches[row_idx]
        patch.set_hatch(BEST_BAR_HATCH)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.2)
        summary_rows.append(f'{dataset_label}: {best_model} ({best_value:.2f}%)')

    return summary_rows


def add_best_summary(fig, title, rows):
    fig.text(
        0.02,
        0.02,
        title + '\\n' + '\\n'.join(rows),
        ha='left',
        va='bottom',
        fontsize=8.5,
        bbox={'boxstyle': 'round,pad=0.35', 'facecolor': 'white', 'edgecolor': '0.6', 'alpha': 0.95},
    )


fig, ax = plt.subplots(figsize=(14, 6))
plot_nrmse.plot(
    kind='bar',
    yerr=plot_stddev,
    ax=ax,
    width=0.82,
    capsize=2,
    error_kw={'elinewidth': 0.8, 'alpha': 0.75},
)
best_nrmse_rows = highlight_dataset_best(ax, plot_nrmse)
ax.set_ylabel('Test nRMSE (%)')
ax.set_xlabel('')
ax.set_title('PyNNLF model performance grouped by SA BESS dataset, with stddev whiskers')
ax.tick_params(axis='x', rotation=0)
handles, labels = ax.get_legend_handles_labels()
handles.append(Patch(facecolor='white', edgecolor='black', hatch=BEST_BAR_HATCH, label='Best within dataset'))
labels.append('Best within dataset')
ax.legend(
    handles,
    labels,
    title='Model/hyperparameter',
    bbox_to_anchor=(1.01, 1.0),
    loc='upper left',
    fontsize=8,
)
add_best_summary(fig, 'Best within dataset (lowest test nRMSE):', best_nrmse_rows)
fig.tight_layout(rect=(0, 0.16, 0.82, 1))
save_figure(fig, 'fig4_model_performance_by_dataset.png')
plt.show()

fig, ax = plt.subplots(figsize=(14, 6))
plot_stddev.plot(
    kind='bar',
    ax=ax,
    width=0.82,
)
best_stddev_rows = highlight_dataset_best(ax, plot_stddev)
ax.set_ylabel('Test nRMSE stddev (%)')
ax.set_xlabel('')
ax.set_title('PyNNLF test nRMSE stddev grouped by SA BESS dataset')
ax.tick_params(axis='x', rotation=0)
handles, labels = ax.get_legend_handles_labels()
handles.append(Patch(facecolor='white', edgecolor='black', hatch=BEST_BAR_HATCH, label='Lowest stddev within dataset'))
labels.append('Lowest stddev within dataset')
ax.legend(
    handles,
    labels,
    title='Model/hyperparameter',
    bbox_to_anchor=(1.01, 1.0),
    loc='upper left',
    fontsize=8,
)
add_best_summary(fig, 'Lowest variability within dataset (test nRMSE stddev):', best_stddev_rows)
fig.tight_layout(rect=(0, 0.16, 0.82, 1))
save_figure(fig, 'fig5_test_nrmse_stddev_by_dataset.png')
plt.show()""",
        ),
        (
            "markdown",
            "## Figure Outputs\n\nThe notebook writes PNG figures to `publication/journal_article_1/results/figures`.",
        ),
        (
            "code",
            """for path in sorted(FIGURES_DIR.glob('fig*.png')):
    print(path)""",
        ),
    ]


def write_publication_notebooks() -> None:
    """Create documented notebooks for diagnostics, processing, validation, result tables, and figures."""

    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)

    diagnostics_nb = notebook(
        [
            ("markdown", "# 1. Diagnose SA BESS Signal Definition\n\nThis notebook runs the physical-plausibility diagnostics used to choose the BESS signal definition."),
            ("markdown", "## Setup\n\nImport the reusable publication workflow helpers."),
            ("code", "from pathlib import Path\nimport sys\n\nPROJECT_DIR = Path.cwd().parents[0] if Path.cwd().name == 'notebooks' else Path.cwd()\nsys.path.insert(0, str(PROJECT_DIR / 'scripts'))\nfrom sa_bess_publication_workflow import run_diagnostics_and_select, REPORT_PATH"),
            ("markdown", "## Run Diagnostics\n\nThis pass scans the raw daily parquet files, scores candidate definitions, and selects the 100-household window."),
            ("code", "selected_profile_id, selected_households, selection_info, metric_summary = run_diagnostics_and_select()\nmetric_summary"),
            ("markdown", "## Report Location\n\nThe Markdown report records the selected profile, evidence table, and chosen window."),
            ("code", "print(REPORT_PATH)"),
        ]
    )
    nbf.write(diagnostics_nb, NOTEBOOK_DIR / "1_diagnose_sa_bess_signal_definition.ipynb")

    processing_nb = notebook(
        [
            ("markdown", "# 2. Process SA BESS Data\n\nThis notebook regenerates the 100-household, one-year BESS aggregate from the selected signal definition."),
            ("markdown", "## Setup\n\nLoad workflow helpers and either reuse diagnostic outputs or rerun diagnostics if needed."),
            ("code", "from pathlib import Path\nimport sys\nimport pandas as pd\n\nPROJECT_DIR = Path.cwd().parents[0] if Path.cwd().name == 'notebooks' else Path.cwd()\nsys.path.insert(0, str(PROJECT_DIR / 'scripts'))\nfrom sa_bess_publication_workflow import run_diagnostics_and_select, process_selected_sample, PROCESSED_DIR"),
            ("markdown", "## Select Profile And Households\n\nThe diagnostics function writes the report and returns the selected profile, selected households, and selected AEST window."),
            ("code", "selected_profile_id, selected_households, selection_info, metric_summary = run_diagnostics_and_select()\nselected_households.head()"),
            ("markdown", "## Export Aggregate And Datasets\n\nThe 5-minute aggregate sums household power. The 30-minute PyNNLF datasets use arithmetic mean power."),
            ("code", "aggregate_5min, site_timeseries, household_summary = process_selected_sample(selected_profile_id, selected_households, selection_info)\nprint(aggregate_5min.shape)\nprint(PROCESSED_DIR)"),
            ("markdown", "## QA Snapshot\n\nCheck row counts, timestamp spacing, and selected household count."),
            ("code", "print('5-min rows:', len(aggregate_5min))\nprint('households:', household_summary['site_id'].nunique())\nprint('start:', aggregate_5min['datetime'].min())\nprint('end:', aggregate_5min['datetime'].max())\nprint('missing cells:', aggregate_5min.isna().sum().sum())"),
        ]
    )
    nbf.write(processing_nb, NOTEBOOK_DIR / "2_process_sa_bess_data.ipynb")

    visual_nb = notebook(
        [
            ("markdown", "# 3. Visualise SA BESS Data\n\nThis notebook provides quick validation plots for the regenerated BESS aggregate."),
            ("markdown", "## Setup\n\nLoad the exported 5-minute and 30-minute aggregate files."),
            ("code", "from pathlib import Path\nimport sys\nimport pandas as pd\nimport matplotlib.pyplot as plt\n\nPROJECT_DIR = Path.cwd().parents[0] if Path.cwd().name == 'notebooks' else Path.cwd()\nsys.path.insert(0, str(PROJECT_DIR / 'scripts'))\nfrom sa_bess_publication_workflow import PROCESSED_DIR\nagg_5 = pd.read_csv(PROCESSED_DIR / 'sa_bess_aggregate_5min.csv', parse_dates=['datetime'])\nagg_30 = pd.read_csv(PROCESSED_DIR / 'sa_bess_aggregate_30min_mean_power.csv', parse_dates=['datetime'])\nagg_5.head()"),
            ("markdown", "## Aggregate Time-Series Check\n\nPlot one representative week for the three target series."),
            ("code", "week = agg_30.iloc[: 48 * 7]\nax = week.set_index('datetime').plot(figsize=(12, 5), linewidth=1.4)\nax.set_ylabel('kW')\nax.set_title('SA BESS aggregate - first selected week')\nplt.show()"),
            ("markdown", "## Distribution Check\n\nInspect summary statistics for the three 30-minute mean-power series."),
            ("code", "agg_30.describe().T"),
        ]
    )
    nbf.write(visual_nb, NOTEBOOK_DIR / "3_visualise_sa_bess_data.ipynb")

    process_output_nb = notebook(
        [
            ("markdown", "# 5. Process PyNNLF Output\n\nThis notebook converts the PyNNLF recap into publication-ready nRMSE comparison tables after the SA BESS PyNNLF batch has finished."),
            ("markdown", "## Setup\n\nImport the result table helper."),
            ("code", "from pathlib import Path\nimport sys\n\nPROJECT_DIR = Path.cwd().parents[0] if Path.cwd().name == 'notebooks' else Path.cwd()\nsys.path.insert(0, str(PROJECT_DIR / 'scripts'))\nfrom process_pynnlf_output import build_publication_tables"),
            ("markdown", "## Build Tables\n\nThe output CSVs contain model/hyperparameter rows and one column for each BESS dataset."),
            ("code", "nrmse_table, nrmse_stddev_table = build_publication_tables(PROJECT_DIR)\ndisplay(nrmse_table)\ndisplay(nrmse_stddev_table)"),
        ]
    )
    nbf.write(process_output_nb, NOTEBOOK_DIR / "5_process_pynnlf_output.ipynb")

    pynnlf_visualisation_nb = notebook(pynnlf_visualisation_cells())
    nbf.write(pynnlf_visualisation_nb, NOTEBOOK_DIR / "6_visualise_pynnlf_sa_bess.ipynb")


def run_processing_workflow() -> None:
    """Run diagnostics, selected-sample processing, dataset export, spec writing, and notebook creation."""

    selected_profile_id, selected_households, selection_info, _ = run_diagnostics_and_select()
    process_selected_sample(selected_profile_id, selected_households, selection_info)
    write_experiment_specs()
    write_publication_notebooks()
    print("[workflow] SA BESS processing workflow complete")


if __name__ == "__main__":
    run_processing_workflow()
