"""Create provenance-labelled RAC-5G prediction log entries.

Prediction logs are produced before field observations are imported.
The city/operator labels describe intended evaluation context, not
evidence that the trained model generalises to those settings.
"""
from datetime import datetime, timezone
import math


def build_prediction_log_entry(prediction, city, operator, generated_at=None):
    if city not in {"Abuja", "Lagos"}:
        raise ValueError("Select a supported city")
    if operator not in {"MTN", "Airtel"} or (city == "Lagos" and operator != "Airtel"):
        raise ValueError("Select an operator within the pilot scope")
    required = ("prediction_id", "prediction", "lower", "upper", "temperature", "pressure", "humidity")
    if any(prediction.get(key) is None for key in required):
        raise ValueError("Prediction record is incomplete")
    numeric = [float(prediction[key]) for key in ("prediction", "lower", "upper", "temperature", "pressure", "humidity")]
    if not all(math.isfinite(v) for v in numeric) or numeric[1] > numeric[2]:
        raise ValueError("Prediction values are invalid")
    stamp = generated_at or datetime.now(timezone.utc)
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError("Prediction timestamp must include timezone")
    return {
        "prediction_id": str(prediction["prediction_id"]),
        "timestamp_utc": stamp.astimezone(timezone.utc).isoformat(),
        "city": city,
        "operator": operator,
        "predicted_rssi_dbm": numeric[0],
        "lower_rssi_dbm": numeric[1],
        "upper_rssi_dbm": numeric[2],
        "provenance": "ClimateNetAI-RAC5G model-generated prediction",
        "temperature_c": numeric[3],
        "pressure_hpa": numeric[4],
        "humidity_pct": numeric[5],
        "climate_source": str(prediction.get("climate_source", "")),
        "rbe_state": str(prediction.get("rbe_state", "")),
        "evaluation_status": "experimental_not_externally_validated",
    }
