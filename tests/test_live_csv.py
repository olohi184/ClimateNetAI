import unittest
from src.live_csv import parse_observation_csv


HEADER = "timestamp,city,operator,network_type,source,metric_type,signal_dbm"
GOOD = "2026-10-13T14:00:00+01:00,Abuja,MTN,5G NR NSA,csv,RSSI,-87"
RSRP = "2026-10-13T14:00:00+01:00,Lagos,Airtel,5G NR SA,csv,NR_SS_RSRP,-100"


class CSVTests(unittest.TestCase):
    def test_mixed_metrics(self):
        accepted, rejected = parse_observation_csv("\\n".join([HEADER, GOOD, RSRP]).replace("\\n", "\n"))
        self.assertEqual(len(accepted), 2)
        self.assertEqual(rejected, [])
        self.assertTrue(accepted[0]["eligible_for_rssi_comparison"])
        self.assertFalse(accepted[1]["eligible_for_rssi_comparison"])

    def test_invalid_row_rejected(self):
        rows = "\n".join([HEADER, GOOD, "bad,Abuja,MTN,5G NR NSA,csv,RSSI,-85"])
        accepted, rejected = parse_observation_csv(rows)
        self.assertEqual(len(accepted), 1)
        self.assertEqual(rejected[0]["csv_row"], 3)

    def test_missing_columns(self):
        with self.assertRaises(ValueError):
            parse_observation_csv("timestamp,city\nnow,Abuja")

    def test_file_limit(self):
        with self.assertRaises(ValueError):
            parse_observation_csv(b"x" * 2_000_001)


if __name__ == "__main__":
    unittest.main()
