import unittest
from src.live_pairing import pair_rssi_observation

OBS = {
    "eligible_for_rssi_comparison": True,
    "timestamp_utc": "2026-10-13T13:10:00+00:00",
    "city": "Abuja", "operator": "MTN",
    "observed_rssi_dbm": -84, "source": "csv",
}
PRED = {
    "timestamp_utc": "2026-10-13T13:00:00+00:00",
    "city": "Abuja", "operator": "MTN",
    "predicted_rssi_dbm": -82,
    "lower_rssi_dbm": -92, "upper_rssi_dbm": -72,
    "prediction_id": "test-1", "provenance": "independent_model_output",
}


class PairingTests(unittest.TestCase):
    def test_valid_pair(self):
        result = pair_rssi_observation(OBS, PRED)
        self.assertEqual(result["Absolute_Error_dB"], 2)
        self.assertEqual(result["Interval_Covered"], 1)

    def test_rsrp_not_eligible(self):
        with self.assertRaises(ValueError):
            pair_rssi_observation({**OBS, "eligible_for_rssi_comparison": False}, PRED)

    def test_city_mismatch(self):
        with self.assertRaises(ValueError):
            pair_rssi_observation(OBS, {**PRED, "city": "Lagos"})

    def test_operator_mismatch(self):
        with self.assertRaises(ValueError):
            pair_rssi_observation(OBS, {**PRED, "operator": "Airtel"})

    def test_stale_measurement(self):
        with self.assertRaises(ValueError):
            pair_rssi_observation({**OBS, "timestamp_utc": "2026-10-13T15:00:00+00:00"}, PRED)

    def test_retrospective_measurement(self):
        with self.assertRaises(ValueError):
            pair_rssi_observation({**OBS, "timestamp_utc": "2026-10-13T12:59:00+00:00"}, PRED)


if __name__ == "__main__":
    unittest.main()
