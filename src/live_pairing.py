"""Conservative pairing of independently logged predictions and field RSSI.

Pairing is a candidate comparison, not proof of external validation.
"""
from datetime import datetime
import math


def _parse_time(value):
    dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError("Pairing timestamps require timezone offsets")
    return dt


def pair_rssi_observation(observation, prediction, max_gap_minutes=30):
    """Match one prospective RSSI observation to a prior prediction.

    Both records require city, operator and time. Prediction also requires
    prediction_id, provenance, and RSSI point and interval estimates.
    """
    if not observation.get("eligible_for_rssi_comparison"):
        raise ValueError("Only measured RSSI can be compared with RSSI predictions")
    for key in ("city", "operator"):
        if not observation.get(key) or observation[key] != prediction.get(key):
            raise ValueError(f"Missing or mismatched {key}")
    if not prediction.get("prediction_id") or not prediction.get("provenance"):
        raise ValueError("Prediction ID and provenance are required")
    if not 0 <= max_gap_minutes <= 60:
        raise ValueError("Pairing window must be 0 to 60 minutes")

    try:
        obs_time = _parse_time(observation["timestamp_utc"])
        pred_time = _parse_time(prediction["timestamp_utc"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid or missing timestamp") from exc
    gap = (obs_time - pred_time).total_seconds() / 60
    if gap < 0:
        raise ValueError("Observation precedes prediction; prospective comparison invalid")
    if gap > max_gap_minutes:
        raise ValueError("Prediction and observation timestamps are too far apart")

    try:
        actual = float(observation["observed_rssi_dbm"])
        predicted = float(prediction["predicted_rssi_dbm"])
        lower = float(prediction["lower_rssi_dbm"])
        upper = float(prediction["upper_rssi_dbm"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid or missing RSSI values") from exc
    if not all(math.isfinite(x) for x in (actual, predicted, lower, upper)):
        raise ValueError("RSSI values must be finite")
    if lower > upper:
        raise ValueError("Invalid prediction interval")
    return {
        "Prediction_ID": str(prediction["prediction_id"]),
        "City": observation["city"],
        "Operator": observation["operator"],
        "Prediction_Time_UTC": pred_time.isoformat(),
        "Observation_Time_UTC": obs_time.isoformat(),
        "Pairing_Gap_Minutes": round(gap, 3),
        "Prediction_Provenance": str(prediction["provenance"]),
        "Observation_Source": observation.get("source", ""),
        "Predicted_RSSI_dBm": predicted,
        "Observed_RSSI_dBm": actual,
        "Absolute_Error_dB": abs(actual - predicted),
        "Interval_Covered": int(lower <= actual <= upper),
        "Validation_Status": "candidate_pair_requires_provenance_review",
    }
