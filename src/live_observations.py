"""Validate external mobile-network observations before RAC-5G ingestion.

This module does not claim to acquire live Android radio telemetry.
It provides a safe boundary for manual, CSV, or future API inputs.
Only explicitly identified RSSI (dBm) can be used as the existing
RAC-5G model's observed target. NR SS-RSRP is NOT interchangeable.
"""
from __future__ import annotations

from datetime import datetime, timezone
import math

ALLOWED_OPERATORS = {"MTN", "Airtel"}
ALLOWED_CITIES = {"Abuja", "Lagos"}
ALLOWED_NETWORKS = {"5G NR SA", "5G NR NSA", "4G LTE", "Unknown"}
ALLOWED_SOURCES = {"manual", "csv", "android_app", "iphone_field_test", "other"}


def _number(value, name, low, high):
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def validate_observation(payload: dict) -> dict:
    """Return a normalized record; reject incompatible or ambiguous inputs.

    Required: timestamp (ISO-8601 with timezone), city, operator,
    network_type, source, metric_type. Metric type is 'RSSI' or
    'NR_SS_RSRP'; measured value is optional for weather-only inference.
    """
    if not isinstance(payload, dict):
        raise ValueError("Observation must be a dictionary")

    required = ("timestamp", "city", "operator", "network_type", "source", "metric_type")
    missing = [k for k in required if not payload.get(k)]
    if missing:
        raise ValueError("Missing fields: " + ", ".join(missing))

    try:
        stamp = datetime.fromisoformat(str(payload["timestamp"]).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("timestamp must be ISO-8601") from exc
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError("timestamp must include a timezone offset")

    city, operator = payload["city"], payload["operator"]
    if city not in ALLOWED_CITIES or operator not in ALLOWED_OPERATORS:
        raise ValueError("Unsupported city or operator")
    if city == "Lagos" and operator != "Airtel":
        raise ValueError("Current pilot scope in Lagos is Airtel only")
    if payload["network_type"] not in ALLOWED_NETWORKS:
        raise ValueError("Unknown network_type")
    if payload["source"] not in ALLOWED_SOURCES:
        raise ValueError("Unknown source")
    if payload["metric_type"] not in {"RSSI", "NR_SS_RSRP"}:
        raise ValueError("metric_type must be RSSI or NR_SS_RSRP")

    value = _number(payload.get("signal_dbm"), "signal_dbm", -160, -10)
    temp = _number(payload.get("temperature_c"), "temperature_c", -20, 60)
    pressure = _number(payload.get("pressure_hpa"), "pressure_hpa", 800, 1100)
    humidity = _number(payload.get("humidity_pct"), "humidity_pct", 0, 100)
    if any(v is not None for v in (temp, pressure, humidity)) and any(
        v is None for v in (temp, pressure, humidity)
    ):
        raise ValueError("Atmospheric inputs must include temperature, pressure and humidity together")

    return {
        "timestamp_utc": stamp.astimezone(timezone.utc).isoformat(),
        "city": city,
        "operator": operator,
        "network_type": payload["network_type"],
        "source": payload["source"],
        "metric_type": payload["metric_type"],
        "signal_dbm": value,
        "observed_rssi_dbm": value if payload["metric_type"] == "RSSI" else None,
        "temperature_c": temp,
        "pressure_hpa": pressure,
        "humidity_pct": humidity,
        "weather_source": str(payload.get("weather_source") or "").strip(),
        "device_model": str(payload.get("device_model") or "").strip(),
        "participant_code": str(payload.get("participant_code") or "").strip(),
        "measurement_context": str(payload.get("measurement_context") or "").strip(),
        "eligible_for_rssi_comparison": payload["metric_type"] == "RSSI" and value is not None,
        "acquisition_mode": "submitted_observation",
    }
