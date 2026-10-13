import unittest
from src.live_observations import validate_observation


BASE = {
    "timestamp": "2026-10-13T14:00:00+01:00",
    "city": "Abuja", "operator": "MTN",
    "network_type": "5G NR NSA", "source": "manual",
    "metric_type": "RSSI", "signal_dbm": -85,
}


class LiveObservationTests(unittest.TestCase):
    def test_rssi_eligible(self):
        row = validate_observation(BASE)
        self.assertTrue(row["eligible_for_rssi_comparison"])
        self.assertEqual(row["observed_rssi_dbm"], -85.0)

    def test_rsrp_not_rssi(self):
        row = validate_observation({**BASE, "metric_type": "NR_SS_RSRP"})
        self.assertFalse(row["eligible_for_rssi_comparison"])
        self.assertIsNone(row["observed_rssi_dbm"])

    def test_naive_timestamp_rejected(self):
        with self.assertRaises(ValueError):
            validate_observation({**BASE, "timestamp": "2026-10-13T14:00:00"})

    def test_partial_weather_rejected(self):
        with self.assertRaises(ValueError):
            validate_observation({**BASE, "temperature_c": 29})

    def test_out_of_scope_operator_rejected(self):
        with self.assertRaises(ValueError):
            validate_observation({**BASE, "city": "Lagos", "operator": "MTN"})

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            validate_observation({**BASE, "signal_dbm": float("nan")})


if __name__ == "__main__":
    unittest.main()
