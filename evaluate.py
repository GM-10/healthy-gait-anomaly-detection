#!/usr/bin/env python3
"""
evaluate.py — Master evaluation entry point for the SIAT-LLMD gait anomaly detection project.

Runs the complete post-hoc evaluation pipeline on scored CSVs produced by the sEMG
and Kinematics+Kinetics pipelines. No re-training or re-inference is performed.

Usage:
    python evaluate.py --config configs/transformer.yaml
    python evaluate.py --config configs/all_models.yaml --dry_run
    python evaluate.py --config configs/lstm.yaml --skip_plots --skip_stats

Prerequisites:
    1. Run semg_pipeline with --save_train_errors:
       python -m semg_pipeline.run_pipeline --save_train_errors [other args]

    2. Run kinetics_pipeline with --save_train_errors:
       python -m kinetics_pipeline.run_pipeline --save_train_errors [other args]

    3. Optionally run late_fusion:
       python fusion/late_fusion.py [args]

Outputs (in output_dir/):
    tables/                   CSV + Markdown + LaTeX comparison tables
    figures/                  PNG, PDF, SVG plots
    statistics/               Bootstrap CI + McNemar tables
    calibration/              Threshold sweep + overlap analysis
    thresholds/               JSON threshold results per model/channel
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Ensure repo root on path
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Framework imports
from evaluation_framework.config import load_config, EvalConfig
from evaluation_framework.thresholds import (
    fit_all_thresholds,
    save_all_thresholds,
    compare_thresholds,
    THRESHOLD_METHODS,
)
from evaluation_framework.evaluation import (
    evaluate_all_thresholds,
    aggregate_by_modality,
    build_comparison_dataframe,
    evaluate_fusion_strategies,
)
from evaluation_framework.statistics import run_statistical_analysis, export_statistics
from evaluation_framework.severity_analysis import SeverityAnalyzer
from evaluation_framework.fusion_analysis import FusionAnalyzer
from evaluation_framework.calibration import CalibrationAnalyzer
from evaluation_framework.plots import plot_all
from evaluation_framework.tables import export_all_tables

# Pipeline-specific train-error loaders
try:
    from semg_pipeline.anomaly_scorer import load_train_errors as load_semg_train_errors
except ImportError:
    load_semg_train_errors = None

try:
    from kinetics_pipeline.anomaly_scorer import load_train_errors as load_kinkin_train_errors
except ImportError:
    load_kinkin_train_errors = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate")


# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

_MODEL_DISPLAY = {
    "lstm":        "LSTM",
    "transformer": "Transformer",
    "sarima":      "SARIMA",
}


def _find_score_csvs(
    root_dir: str,
    subjects: List[str],
    movements: List[str],
    model_keys: List[str],
) -> List[str]:
    """
    Walk root_dir recursively and collect score CSVs for the given subjects,
    movements, and models.

    Score CSVs are expected to contain the model_name and movement in the
    filename or in the data columns. We do a broad glob and filter by content.
    """
    if not os.path.isdir(root_dir):
        logger.warning(f"  Directory not found: {root_dir}")
        return []

    csvs: List[str] = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.endswith("_scores.csv"):
                # Check subject match from directory or filename
                path = os.path.join(dirpath, fname)
                # Subject filter: directory name must contain a subject ID
                if any(s.lower() in dirpath.lower() or s.lower() in fname.lower() for s in subjects):
                    # Movement filter
                    if any(mov.lower() in fname.lower() for mov in movements):
                        csvs.append(path)
                    elif not movements:  # no filter
                        csvs.append(path)
    return sorted(csvs)


def _load_score_csvs(
    csv_paths: List[str],
    model_display_names: List[str],
) -> pd.DataFrame:
    """
    Load and concatenate multiple score CSVs.

    Gracefully handles:
    - Missing optional columns (severity → inferred as 0.35 for anomalies)
    - model_name column not present (inferred from path)
    - Duplicate rows (dropped)

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame ready for evaluate_all_thresholds().
    """
    dfs: List[pd.DataFrame] = []

    for path in csv_paths:
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            logger.warning(f"  Could not load {path}: {exc}")
            continue

        # Filter to requested model display names
        if "model_name" in df.columns:
            df = df[df["model_name"].isin(model_display_names)]
        else:
            # Infer model_name from filename heuristic
            path_lower = path.lower()
            for model_display in model_display_names:
                if model_display.lower() in path_lower:
                    df["model_name"] = model_display
                    break
            else:
                logger.debug(f"  Could not infer model_name from {path} — skipping")
                continue

        if df.empty:
            continue

        # Ensure required columns exist
        if "reconstruction_error" not in df.columns or "is_synthetic_anomaly" not in df.columns:
            logger.warning(f"  Missing required columns in {path} — skipping")
            continue

        # Fill missing optional columns
        if "severity" not in df.columns:
            # Infer from anomaly label: moderate severity (0.35) for all anomalies
            df["severity"] = df["is_synthetic_anomaly"].astype(float) * 0.35

        if "anomaly_type" not in df.columns:
            df["anomaly_type"] = "unknown"

        if "movement" not in df.columns:
            # Infer from filename
            fname = os.path.basename(path)
            for token in fname.split("_"):
                df["movement"] = token  # rough approximation
                break

        dfs.append(df)

    if not dfs:
        logger.warning("  No valid score CSVs loaded.")
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates().reset_index(drop=True)
    logger.info(f"  Loaded {len(combined):,} rows from {len(csv_paths)} CSVs")
    return combined


def _load_all_train_errors(
    config: EvalConfig,
    model_display_names: List[str],
    semg_channels: Optional[List[str]],
    kinkin_channels: Optional[List[str]],
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load persisted training errors for all models × channels.

    Returns
    -------
    Dict[str, Dict[str, np.ndarray]]
        {model_display_name: {channel_name: np.ndarray}}
    """
    all_errors: Dict[str, Dict[str, np.ndarray]] = {}

    for model_display in model_display_names:
        ch_errors: Dict[str, np.ndarray] = {}

        # sEMG channels
        if load_semg_train_errors is not None and semg_channels:
            semg_ch = load_semg_train_errors(
                model_name=model_display,
                channel_names=semg_channels,
                output_dir=config.semg_dir,
            )
            ch_errors.update(semg_ch)

        # KinKin channels
        if load_kinkin_train_errors is not None and kinkin_channels:
            kinkin_ch = load_kinkin_train_errors(
                model_name=model_display,
                channel_names=kinkin_channels,
                output_dir=config.kinkin_dir,
            )
            ch_errors.update(kinkin_ch)

        if ch_errors:
            all_errors[model_display] = ch_errors
        else:
            logger.warning(
                f"  [TrainErrors] No .npy files found for {model_display}. "
                "Run pipelines with --save_train_errors. "
                "Falling back to test-set clean windows for threshold computation."
            )

    return all_errors


def _compute_all_thresholds(
    score_df: pd.DataFrame,
    train_errors_map: Dict[str, Dict[str, np.ndarray]],
    methods: List[str],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Compute all threshold methods for every (model, channel).

    Returns
    -------
    {model_display_name: {channel_name: {method: ThresholdResult}}}
    """
    results: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for model_name in score_df["model_name"].unique():
        results[model_name] = {}
        ch_map = train_errors_map.get(model_name, {})

        for ch_name in sorted(score_df[score_df["model_name"] == model_name]["channel_name"].unique()):
            train_errors = ch_map.get(ch_name)

            if train_errors is None or len(train_errors) == 0:
                # Fallback: use clean test windows
                ch_df = score_df[
                    (score_df["model_name"] == model_name) &
                    (score_df["channel_name"] == ch_name) &
                    (score_df["is_synthetic_anomaly"] == 0)
                ]
                if ch_df.empty:
                    continue
                train_errors = ch_df["reconstruction_error"].values.astype(float)

            logger.info(f"  [Threshold] {model_name}/{ch_name}: fitting {methods} …")
            results[model_name][ch_name] = fit_all_thresholds(train_errors, methods)

    return results


def _load_fusion_df(fusion_dir: str, model_display_names: List[str]) -> pd.DataFrame:
    """
    Attempt to load fused output CSVs from the late_fusion step.
    """
    dfs: List[pd.DataFrame] = []
    for dirpath, _, fnames in os.walk(fusion_dir):
        for fname in fnames:
            if "fused" in fname.lower() and fname.endswith(".csv"):
                try:
                    df = pd.read_csv(os.path.join(dirpath, fname))
                    if "model_name" in df.columns:
                        df = df[df["model_name"].isin(model_display_names)]
                    dfs.append(df)
                except Exception:
                    pass
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(
    config: EvalConfig,
    skip_plots: bool = False,
    skip_stats: bool = False,
    skip_fusion: bool = False,
) -> None:
    """
    Run the full evaluation pipeline.

    Parameters
    ----------
    config : EvalConfig
        Loaded and validated configuration.
    skip_plots : bool
        Skip figure generation (useful for fast iterations).
    skip_stats : bool
        Skip bootstrap / McNemar tests (fast, for debugging).
    skip_fusion : bool
        Skip fusion analysis (when fusion CSVs not yet generated).
    """
    t0 = time.time()
    os.makedirs(config.output_dir, exist_ok=True)

    model_display_names = [_MODEL_DISPLAY[m] for m in config.models if m in _MODEL_DISPLAY]

    logger.info("=" * 65)
    logger.info("  Evaluation Framework — SIAT-LLMD Gait Anomaly Detection")
    logger.info("=" * 65)
    config.log()

    # ── Step 1: Discover + load score CSVs ───────────────────────────────────
    logger.info("\n[1/9] Loading score CSVs …")

    semg_csvs   = _find_score_csvs(config.semg_dir,   config.test_subjects, config.movements, config.models)
    kinkin_csvs = _find_score_csvs(config.kinkin_dir, config.test_subjects, config.movements, config.models)
    all_csvs    = semg_csvs + kinkin_csvs

    if not all_csvs:
        logger.error(
            f"No score CSVs found under {config.semg_dir} or {config.kinkin_dir}.\n"
            "Run the sEMG and kinetics pipelines first, then re-run evaluate.py."
        )
        sys.exit(1)

    logger.info(f"  Found {len(semg_csvs)} sEMG CSVs + {len(kinkin_csvs)} KinKin CSVs")
    score_df = _load_score_csvs(all_csvs, model_display_names)

    if score_df.empty:
        logger.error("No valid rows loaded from score CSVs — exiting.")
        sys.exit(1)

    # ── Step 2: Load training errors ─────────────────────────────────────────
    logger.info("\n[2/9] Loading training reconstruction errors …")

    semg_channels   = sorted(score_df[score_df["modality"] == "sEMG"]["channel_name"].unique().tolist()) if "modality" in score_df.columns else []
    kinkin_channels = sorted(score_df[score_df["modality"] == "Kinematics+Kinetics"]["channel_name"].unique().tolist()) if "modality" in score_df.columns else []

    train_errors_map = _load_all_train_errors(
        config, model_display_names, semg_channels, kinkin_channels
    )

    # ── Step 3: Compute all thresholds ───────────────────────────────────────
    logger.info(f"\n[3/9] Fitting thresholds: {config.threshold_methods} …")
    threshold_results_map = _compute_all_thresholds(
        score_df, train_errors_map, config.threshold_methods
    )
    save_all_thresholds(threshold_results_map, config.output_dir)

    # ── Step 3b: Threshold comparison table ──────────────────────────────────
    logger.info("  [3b] Comparing threshold methods (value, rel-diff, label-agreement) …")
    threshold_comparison_df = compare_thresholds(threshold_results_map, score_df)
    thresh_comp_csv = os.path.join(config.output_dir, "threshold_comparison.csv")
    threshold_comparison_df.to_csv(thresh_comp_csv, index=False)
    logger.info(f"  Threshold comparison → {thresh_comp_csv}")

    # ── Step 4: Evaluate all thresholds ──────────────────────────────────────
    logger.info("\n[4/9] Computing metrics for all (model × channel × threshold) …")
    metrics_df = evaluate_all_thresholds(
        score_df=score_df,
        train_errors_map=train_errors_map,
        methods=config.threshold_methods,
    )

    aggregate_df   = aggregate_by_modality(metrics_df)
    comparison_df  = build_comparison_dataframe(metrics_df, aggregate_df)

    # Save raw metrics
    metrics_path = os.path.join(config.output_dir, "metrics_all.csv")
    comparison_df.to_csv(metrics_path, index=False)
    logger.info(f"  Metrics → {metrics_path} ({len(comparison_df)} rows)")

    # ── Step 5: Statistical analysis ─────────────────────────────────────────
    stats_report = None
    if not skip_stats:
        logger.info(
            f"\n[5/9] Statistical analysis (n_bootstrap={config.n_bootstrap}, "
            f"seed={config.random_seed}) …"
        )
        stats_report = run_statistical_analysis(
            metrics_df=metrics_df,
            score_df=score_df,
            n_bootstrap=config.n_bootstrap,
            ci_level=config.ci_level,
            seed=config.random_seed,
        )
        stats_dir = os.path.join(config.output_dir, "statistics")
        export_statistics(stats_report, stats_dir)
        logger.info(f"  Statistics → {stats_dir}")
    else:
        logger.info("\n[5/9] Statistical analysis — SKIPPED (--skip_stats)")

    # ── Step 6: Severity analysis ─────────────────────────────────────────────
    logger.info("\n[6/9] Running severity-stratified analysis …")
    severity_analyzer = SeverityAnalyzer(threshold_results_map)
    severity_report   = severity_analyzer.analyze(score_df, methods=config.threshold_methods)

    sev_dir = os.path.join(config.output_dir, "severity")
    severity_paths = severity_analyzer.export(severity_report, sev_dir)
    logger.info(f"  Severity analysis → {sev_dir} ({len(severity_paths)} files)")

    # Severity profile plots (line plots with CI error bars)
    if not skip_plots:
        severity_analyzer.plot_severity_profiles(
            severity_report, sev_dir, formats=config.figure_formats
        )

    if not severity_report.cause_df.empty:
        n_nonmono = len(severity_report.cause_df)
        logger.info(
            f"  ⚠ Non-monotonic severity cases found: {n_nonmono}. "
            f"See {sev_dir}/severity_nonmonotonic_causes.csv"
        )

    # ── Step 7: Fusion analysis ───────────────────────────────────────────────
    fusion_report = None
    fusion_metrics_df = pd.DataFrame()

    if not skip_fusion:
        logger.info("\n[7/9] Running fusion analysis …")
        fusion_dir = os.path.join(os.path.dirname(config.semg_dir), "fusion")
        fused_df = _load_fusion_df(fusion_dir, model_display_names)

        if not fused_df.empty:
            fusion_analyzer = FusionAnalyzer()
            fusion_report   = fusion_analyzer.analyze(fused_df, score_df)

            fusion_out_dir = os.path.join(config.output_dir, "fusion")
            fusion_analyzer.export(fusion_report, fusion_out_dir)
            fusion_analyzer.plot_all(fusion_report, fusion_out_dir, config.figure_formats)

            if "fused_OR" in fused_df.columns and "ground_truth" in fused_df.columns:
                for model_name in model_display_names:
                    model_fused = fused_df[fused_df["model_name"] == model_name] if "model_name" in fused_df.columns else fused_df
                    if model_fused.empty:
                        continue
                    for method in config.threshold_methods:
                        f_metrics = evaluate_fusion_strategies(model_fused, model_name, method)
                        fusion_metrics_df = pd.concat([fusion_metrics_df, f_metrics], ignore_index=True)

            logger.info(f"  Fusion analysis → {fusion_out_dir}")
        else:
            logger.warning(
                "  Fusion CSVs not found — skipping fusion analysis. "
                f"Run fusion/late_fusion.py first (expected under {fusion_dir}/)."
            )
    else:
        logger.info("\n[7/9] Fusion analysis — SKIPPED (--skip_fusion)")

    # ── Step 8: Calibration analysis ─────────────────────────────────────────
    logger.info("\n[8/9] Running calibration analysis …")
    calib_analyzer = CalibrationAnalyzer(threshold_results_map)
    calib_report   = calib_analyzer.analyze(score_df, methods=config.threshold_methods)

    calib_dir = os.path.join(config.output_dir, "calibration")
    calib_analyzer.export(calib_report, calib_dir)
    if not skip_plots:
        calib_analyzer.plot_all(calib_report, calib_dir, config.figure_formats)
    logger.info(f"  Calibration → {calib_dir}")

    # ── Step 9: Export tables and figures ─────────────────────────────────────
    logger.info("\n[9/9] Exporting tables and figures …")

    tables_dir = os.path.join(config.output_dir, "tables")
    table_paths = export_all_tables(
        output_dir=tables_dir,
        comparison_df=comparison_df,
        stats_report=stats_report,
        severity_report=severity_report,
        fusion_report=fusion_report,
        fusion_metrics_df=fusion_metrics_df if not fusion_metrics_df.empty else None,
        threshold_comparison_df=threshold_comparison_df,
    )
    logger.info(f"  Tables → {tables_dir} ({len(table_paths)} files)")

    if not skip_plots:
        fig_dir = os.path.join(config.output_dir, "figures")
        plot_all(
            score_df=score_df,
            metrics_df=metrics_df,
            severity_df=severity_report.aggregate_df if severity_report else None,
            train_errors_map=train_errors_map,
            threshold_results_map=threshold_results_map,
            output_dir=config.output_dir,
            formats=config.figure_formats,
        )
        logger.info(f"  Figures → {fig_dir}")
    else:
        logger.info("  Figures — SKIPPED (--skip_plots)")

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    _print_summary(comparison_df, stats_report, severity_report, config, elapsed)


def _print_summary(comparison_df, stats_report, severity_report, config, elapsed):
    """Print a concise summary to stdout."""
    logger.info("\n" + "=" * 65)
    logger.info("  EVALUATION COMPLETE")
    logger.info("=" * 65)
    logger.info(f"  Output directory : {config.output_dir}")
    logger.info(f"  Elapsed time     : {elapsed:.1f}s")
    logger.info(f"  Total rows       : {len(comparison_df)}")

    # Best F1 per (model, threshold_method)
    if not comparison_df.empty and "f1" in comparison_df.columns:
        agg_rows = comparison_df[comparison_df.get("channel_name", pd.Series()) == "ALL_CHANNELS"]
        if agg_rows.empty:
            agg_rows = comparison_df
        logger.info("\n  Best F1 Scores:")
        for (model, method), grp in agg_rows.groupby(["model_name", "threshold_method"]):
            best_f1 = grp["f1"].max()
            logger.info(f"    {model:15s}  {method:15s}  F1={best_f1:.4f}")

    if stats_report is not None and not stats_report.mcnemar_df.empty:
        n_sig = stats_report.mcnemar_df["mcnemar_sig"].sum()
        logger.info(f"\n  McNemar significant pairs : {n_sig} / {len(stats_report.mcnemar_df)}")

    if severity_report is not None and not severity_report.cause_df.empty:
        logger.info(f"  Non-monotonic cases        : {len(severity_report.cause_df)}")

    logger.info("\n  Tables  → tables/*.{csv,md,tex}")
    logger.info("  Figures → figures/**/*.{png,pdf,svg}")
    logger.info("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-hoc evaluation framework for gait anomaly detection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file (e.g. configs/transformer.yaml).",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Override output_dir from config.",
    )
    parser.add_argument(
        "--threshold_methods",
        nargs="+",
        default=None,
        help="Override threshold methods (e.g. --threshold_methods mean_std percentile95).",
    )
    parser.add_argument(
        "--skip_plots",
        action="store_true",
        help="Skip figure generation.",
    )
    parser.add_argument(
        "--skip_stats",
        action="store_true",
        help="Skip bootstrap CI and McNemar tests.",
    )
    parser.add_argument(
        "--skip_fusion",
        action="store_true",
        help="Skip fusion analysis (when fusion CSVs not yet generated).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help=(
            "Dry run: restrict to Sub36 / WAK / moderate / 100 bootstrap samples. "
            "Use for smoke-testing without a full dataset."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Build CLI overrides
    overrides = {}
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    if args.threshold_methods:
        overrides["threshold_methods"] = args.threshold_methods
    if args.dry_run:
        overrides["dry_run"] = True

    # Load config
    config = load_config(args.config, overrides=overrides)

    # Run evaluation
    run_evaluation(
        config=config,
        skip_plots=args.skip_plots,
        skip_stats=args.skip_stats,
        skip_fusion=args.skip_fusion,
    )


if __name__ == "__main__":
    main()
