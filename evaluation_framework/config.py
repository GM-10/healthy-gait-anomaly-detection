"""
evaluation_framework/config.py

Configuration loading and validation for the evaluation framework.

Supports YAML config files with the schema:

    threshold_methods: [mean_std, percentile95, percentile99]
    models: [lstm, transformer]
    n_bootstrap: 1000
    ci_level: 0.95
    random_seed: 42
    output_dir: outputs/evaluation
    semg_dir: outputs/sEMG
    kinkin_dir: outputs/kinetics
    figure_formats: [png, pdf, svg]
    movements: [WAK, UPS, SITDN]
    test_subjects: [Sub36, Sub37, Sub38, Sub39, Sub40]
    severity_levels: [mild, moderate, severe]

CLI override:
    python evaluate.py --config configs/transformer.yaml --output_dir /custom/dir
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_MOVEMENTS = [
    "WAK", "UPS", "DNS", "HS", "KLCL", "KLFT", "LLB", "LLF",
    "LLS", "LUGB", "LUGF", "SITDN", "STC", "STDUP", "TO", "TPTO",
]

_DEFAULT_TEST_SUBS = [f"Sub{i:02d}" for i in range(36, 41)]

_VALID_THRESHOLD_METHODS = {"mean_std", "percentile95", "percentile99"}
_VALID_MODELS           = {"lstm", "transformer", "sarima"}
_VALID_FIG_FORMATS      = {"png", "pdf", "svg"}
_VALID_SEVERITIES       = {"mild", "moderate", "severe"}


# ─────────────────────────────────────────────────────────────────────────────
# Config dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalConfig:
    """Validated evaluation configuration."""

    # Thresholding
    threshold_methods: List[str] = field(
        default_factory=lambda: ["mean_std", "percentile95", "percentile99"]
    )

    # Models to evaluate
    models: List[str] = field(default_factory=lambda: ["lstm", "transformer"])

    # Statistical analysis
    n_bootstrap:   int   = 1000
    ci_level:      float = 0.95
    random_seed:   int   = 42

    # Paths
    output_dir:  str = os.path.join("outputs", "evaluation")
    semg_dir:    str = os.path.join("outputs", "sEMG")
    kinkin_dir:  str = os.path.join("outputs", "kinetics")

    # Experiment scope
    movements:       List[str] = field(default_factory=lambda: list(_DEFAULT_MOVEMENTS))
    test_subjects:   List[str] = field(default_factory=lambda: list(_DEFAULT_TEST_SUBS))
    severity_levels: List[str] = field(default_factory=lambda: ["mild", "moderate", "severe"])

    # Output formats
    figure_formats: List[str] = field(default_factory=lambda: ["png", "pdf", "svg"])

    # Dry-run: restrict to one subject for smoke-testing
    dry_run: bool = False

    def validate(self) -> None:
        """Raise ValueError for invalid config values."""
        invalid_thresh = set(self.threshold_methods) - _VALID_THRESHOLD_METHODS
        if invalid_thresh:
            raise ValueError(
                f"Unknown threshold_methods: {invalid_thresh}. "
                f"Valid: {_VALID_THRESHOLD_METHODS}"
            )

        invalid_models = set(self.models) - _VALID_MODELS
        if invalid_models:
            raise ValueError(
                f"Unknown models: {invalid_models}. Valid: {_VALID_MODELS}"
            )

        invalid_fmts = set(self.figure_formats) - _VALID_FIG_FORMATS
        if invalid_fmts:
            raise ValueError(
                f"Unknown figure_formats: {invalid_fmts}. Valid: {_VALID_FIG_FORMATS}"
            )

        invalid_sevs = set(self.severity_levels) - _VALID_SEVERITIES
        if invalid_sevs:
            raise ValueError(
                f"Unknown severity_levels: {invalid_sevs}. Valid: {_VALID_SEVERITIES}"
            )

        if not (0.5 < self.ci_level < 1.0):
            raise ValueError(f"ci_level must be in (0.5, 1.0), got {self.ci_level}")

        if self.n_bootstrap < 100:
            raise ValueError(f"n_bootstrap should be >= 100 for reliable CIs, got {self.n_bootstrap}")

    def apply_dry_run(self) -> None:
        """Restrict evaluation scope for smoke-testing."""
        if self.dry_run:
            self.test_subjects   = ["Sub36"]
            self.movements       = ["WAK"]
            self.severity_levels = ["moderate"]
            self.n_bootstrap     = 100
            logger.info("[Config] DRY RUN: restricting to Sub36 / WAK / moderate / 100 bootstrap samples")

    def log(self) -> None:
        """Log the full configuration."""
        logger.info("=" * 60)
        logger.info("  Evaluation Framework Configuration")
        logger.info("=" * 60)
        logger.info(f"  threshold_methods : {self.threshold_methods}")
        logger.info(f"  models            : {self.models}")
        logger.info(f"  n_bootstrap       : {self.n_bootstrap}")
        logger.info(f"  ci_level          : {self.ci_level}")
        logger.info(f"  random_seed       : {self.random_seed}")
        logger.info(f"  output_dir        : {self.output_dir}")
        logger.info(f"  semg_dir          : {self.semg_dir}")
        logger.info(f"  kinkin_dir        : {self.kinkin_dir}")
        logger.info(f"  movements         : {self.movements}")
        logger.info(f"  test_subjects     : {self.test_subjects}")
        logger.info(f"  severity_levels   : {self.severity_levels}")
        logger.info(f"  figure_formats    : {self.figure_formats}")
        logger.info(f"  dry_run           : {self.dry_run}")
        logger.info("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────────────────────

def load_config(yaml_path: str, overrides: Optional[dict] = None) -> EvalConfig:
    """
    Load an EvalConfig from a YAML file with optional CLI overrides.

    Parameters
    ----------
    yaml_path : str
        Path to the YAML config file.
    overrides : dict, optional
        Key-value pairs to override after loading. Typically from argparse.

    Returns
    -------
    EvalConfig
        Validated configuration object.
    """
    try:
        import yaml  # type: ignore
    except ImportError:
        raise ImportError(
            "PyYAML is required to load config files. Install with: pip install pyyaml"
        )

    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f) or {}

    # Apply overrides (CLI args take precedence over YAML)
    if overrides:
        for k, v in overrides.items():
            if v is not None:
                raw[k] = v

    # Build config — only pass keys that EvalConfig knows about
    known_fields = EvalConfig.__dataclass_fields__.keys()
    filtered = {k: v for k, v in raw.items() if k in known_fields}
    unknown   = set(raw.keys()) - set(known_fields)
    if unknown:
        logger.warning(f"[Config] Unknown config keys (ignored): {unknown}")

    cfg = EvalConfig(**filtered)
    cfg.validate()
    cfg.apply_dry_run()
    return cfg


def default_config() -> EvalConfig:
    """Return a default EvalConfig with all defaults filled in."""
    cfg = EvalConfig()
    cfg.validate()
    return cfg
