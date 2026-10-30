"""Reliability Boundary Engine for ClimateNetAI-RAC 5G.

The RBE assigns a prospective selective-prediction state from completed
telemetry only. The current prediction outcome must never be supplied to
this function before its state has been assigned.
"""

import math

RBE_WINDOW = 5
RBE_ERROR_THRESHOLD_DB = 8.71
RBE_COVERAGE_FLOOR = 0.80
RBE_HALF_WIDTH_DB = 10.11


def compute_rbe_state(telemetry_records):
    """Return WARM-UP, TRUST, CAUTION, or ABSTAIN from prior telemetry."""
    n_prior = len(telemetry_records)
    if n_prior < RBE_WINDOW:
        return {
            "state": "WARM-UP",
            "mae": math.nan,
            "coverage": math.nan,
            "n_prior": n_prior,
        }

    recent = telemetry_records[-RBE_WINDOW:]
    mae = sum(float(row["Absolute_Error_dB"]) for row in recent) / RBE_WINDOW
    coverage = sum(float(row["Interval_Covered"]) for row in recent) / RBE_WINDOW

    error_fail = mae > RBE_ERROR_THRESHOLD_DB
    coverage_fail = coverage < RBE_COVERAGE_FLOOR

    if error_fail and coverage_fail:
        state = "ABSTAIN"
    elif error_fail or coverage_fail:
        state = "CAUTION"
    else:
        state = "TRUST"

    return {
        "state": state,
        "mae": mae,
        "coverage": coverage,
        "n_prior": RBE_WINDOW,
    }
