"""Pilot measurement intake for ClimateNetAI-RAC 5G.

Run as a Streamlit multipage page. This is not direct Android acquisition
and does not perform a model prediction; it validates external observations
for later pairing with predictions.
"""
import csv
from datetime import datetime
from io import StringIO

import pandas as pd
import streamlit as st

from src.live_observations import validate_observation

st.title("📲 Mobile 5G Measurement Intake")
st.caption("Research pilot · Manual or CSV observations · Not an automated phone connection")
st.info(
    "This page validates measurements for future pairing with RAC-5G predictions. "
    "It does not collect data directly from a phone or establish live prediction accuracy. "
    "Do not enter a phone number, IMEI, or precise GPS coordinates."
)

if "mobile_intake_records" not in st.session_state:
    st.session_state.mobile_intake_records = []

with st.form("mobile_intake_form"):
    st.subheader("Submit one observation")
    timestamp = st.text_input(
        "Measurement time (ISO-8601 with timezone)",
        value=datetime.now().astimezone().isoformat(timespec="seconds"),
        help="For example 2026-10-13T14:00:00+01:00",
    )
    a, b = st.columns(2)
    with a:
        city = st.selectbox("City", ["Abuja", "Lagos"])
        operator = st.selectbox("Operator", ["MTN", "Airtel"])
        network_type = st.selectbox("Radio technology", ["Unknown", "5G NR NSA", "5G NR SA", "4G LTE"])
        source = st.selectbox("Measurement source", ["manual", "android_app", "iphone_field_test", "other"])
    with b:
        metric_type = st.selectbox("Measured metric", ["RSSI", "NR_SS_RSRP"])
        signal_text = st.text_input("Signal value (dBm), optional", placeholder="-85")
        device_model = st.text_input("Device model (optional)")
        participant_code = st.text_input("Anonymous participant code (optional)")
    with st.expander("Optional matched atmospheric inputs"):
        temp_text = st.text_input("Temperature (°C)")
        pressure_text = st.text_input("Surface pressure (hPa)")
        humidity_text = st.text_input("Relative humidity (%)")
        weather_source = st.text_input("Weather source and observation time")
    submitted = st.form_submit_button("Validate and add observation")

if submitted:
    payload = {
        "timestamp": timestamp, "city": city, "operator": operator,
        "network_type": network_type, "source": source,
        "metric_type": metric_type, "signal_dbm": signal_text,
        "device_model": device_model, "participant_code": participant_code,
        "temperature_c": temp_text, "pressure_hpa": pressure_text,
        "humidity_pct": humidity_text, "weather_source": weather_source,
    }
    try:
        normalized = validate_observation(payload)
    except ValueError as exc:
        st.error(str(exc))
    else:
        st.session_state.mobile_intake_records.append(normalized)
        st.success("Observation validated and added to this session.")

st.subheader("Batch CSV validation")
st.caption(
    "CSV columns: timestamp, city, operator, network_type, source, metric_type, "
    "signal_dbm; optional device_model, participant_code, temperature_c, "
    "pressure_hpa, humidity_pct, weather_source."
)
upload = st.file_uploader("Choose a CSV file", type=["csv"], key="mobile_intake_upload")
if upload is not None and st.button("Validate CSV rows"):
    try:
        raw = upload.getvalue().decode("utf-8-sig")
        rows = list(csv.DictReader(StringIO(raw)))
        if not rows or len(rows) > 1000:
            raise ValueError("CSV must contain 1–1000 data rows.")
        validated = []
        for index, row in enumerate(rows, start=2):
            try:
                validated.append(validate_observation(row))
            except ValueError as exc:
                raise ValueError(f"CSV row {index}: {exc}") from exc
    except (UnicodeDecodeError, ValueError, csv.Error) as exc:
        st.error(f"CSV rejected: {exc}")
    else:
        st.session_state.mobile_intake_records.extend(validated)
        st.success(f"Validated {len(validated)} CSV observations.")

records = st.session_state.mobile_intake_records
if records:
    df = pd.DataFrame(records)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download validated pilot observations",
        df.to_csv(index=False).encode("utf-8"),
        file_name="climatenetai_external_observations.csv",
        mime="text/csv",
    )
    st.warning(
        "Data are stored in the current Streamlit session, not in a persistent database. "
        "Download the CSV before ending the session. "
        "NR SS-RSRP entries must not be used as observed RSSI."
    )
