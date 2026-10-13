"""CSV ingestion adapter for independently submitted mobile observations.

This does not stream radio readings or make predictions. Rows are validated
through live_observations before they can be considered for RSSI comparison.
"""
import csv
import io

from src.live_observations import validate_observation


MAX_UPLOAD_BYTES = 2_000_000
MAX_ROWS = 1000


def parse_observation_csv(content):
    """Return (accepted_records, rejected_rows) from a UTF-8 CSV.

    Rejected rows contain row number and a reason, not submitted personal data.
    """
    if isinstance(content, bytes):
        if len(content) > MAX_UPLOAD_BYTES:
            raise ValueError("CSV exceeds 2 MB limit")
        content = content.decode("utf-8-sig")
    if not isinstance(content, str):
        raise ValueError("CSV must be UTF-8 text or bytes")
    if len(content.encode("utf-8")) > MAX_UPLOAD_BYTES:
        raise ValueError("CSV exceeds 2 MB limit")
    reader = csv.DictReader(io.StringIO(content))
    if not reader.fieldnames:
        raise ValueError("CSV has no header")
    required = {"timestamp", "city", "operator", "network_type", "source", "metric_type"}
    missing = required.difference(reader.fieldnames)
    if missing:
        raise ValueError("Missing CSV columns: " + ", ".join(sorted(missing)))

    accepted, rejected = [], []
    for number, row in enumerate(reader, start=2):
        if number > MAX_ROWS + 1:
            raise ValueError("CSV exceeds 1000 data rows")
        try:
            accepted.append(validate_observation(row))
        except ValueError as exc:
            rejected.append({"csv_row": number, "reason": str(exc)})
    return accepted, rejected
