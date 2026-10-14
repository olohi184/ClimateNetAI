import unittest
from datetime import datetime, timezone
from src.prediction_logging import build_prediction_log_entry

PRED = {
    "prediction_id": 3, "prediction": -80, "lower": -90,
    "upper": -70, "temperature": 30, "pressure": 1013,
    "humidity": 60, "climate_source": "Manual", "rbe_state": "WARM-UP",
}


class PredictionLoggingTests(unittest.TestCase):
    def test_log(self):
        row = build_prediction_log_entry(PRED, "Abuja", "MTN", datetime(2026, 10, 14, tzinfo=timezone.utc))
        self.assertEqual(row["predicted_rssi_dbm"], -80)
        self.assertEqual(row["timestamp_utc"], "2026-10-14T00:00:00+00:00")

    def test_lagos_scope(self):
        with self.assertRaises(ValueError):
            build_prediction_log_entry(PRED, "Lagos", "MTN")

    def test_invalid_interval(self):
        with self.assertRaises(ValueError):
            build_prediction_log_entry({**PRED, "lower": -60}, "Abuja", "MTN")


if __name__ == "__main__":
    unittest.main()
