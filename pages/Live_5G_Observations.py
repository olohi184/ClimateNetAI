"""Research pilot: review independently submitted mobile measurements.

This page intentionally does not claim direct Android acquisition or
validated live RSSI accuracy. The existing RAC-5G prediction/telemetry
workflow remains in app.py; this page is an ingestion and QA staging area.
"""
import pandas as pd
import streamlit as st

from src.live_csv import parse_observation_csv
from src.live_pairing import pair_rssi_observation
import csv
import io

st.set_page_config(page_title="5G Observation Import | ClimateNetAI", page_icon="📡", layout="wide")
st.title("📡 External 5G Observation Import")
st.caption("ClimateNetAI-RAC 5G · research pilot · manual/CSV acquisition (not continuous device streaming)")
st.info(
    "Upload measurements from consenting participants. This screen validates and "
    "reviews records only; it does not automatically score a prediction, update "
    "RBE reliability states, or establish external predictive accuracy."
)

with st.expander("Required CSV columns and metric definitions", expanded=True):
    st.markdown(
        "Required: `timestamp` (ISO-8601 including timezone), `city` "
        "(Abuja or Lagos), `operator` (MTN or Airtel), `network_type` "
        "(5G NR SA, 5G NR NSA, 4G LTE or Unknown), `source` "
        "(manual, csv, android_app, iphone_field_test or other), "
        "`metric_type` (RSSI or NR_SS_RSRP). Optional: `signal_dbm`, "
        "`temperature_c`, `pressure_hpa`, `humidity_pct`, "
        "`weather_source`, `device_model`, `participant_code`, "
        "`measurement_context`. The current pilot supports Abuja MTN, "
        "Abuja Airtel and Lagos Airtel. No IMEI, phone numbers or precise GPS."
    )
    st.warning(
        "RSSI and NR SS-RSRP are different measurements. NR SS-RSRP is never "
        "converted into RSSI. A 5G icon alone does not prove a 5G NR radio measurement."
    )

sample = (
    "timestamp,city,operator,network_type,source,metric_type,signal_dbm,"
    "device_model,participant_code,measurement_context\n"
    "2026-10-13T14:00:00+01:00,Abuja,MTN,5G NR NSA,csv,RSSI,-85,"
    "Example Android,P001,stationary outdoors\n"
    "2026-10-13T14:05:00+01:00,Lagos,Airtel,5G NR SA,csv,NR_SS_RSRP,-100,"
    "Example Android,P002,stationary indoors\n"
)
st.download_button(
    "⬇️ Download example CSV (illustrative, not empirical data)",
    sample,
    file_name="climatenetai_example_observations.csv",
    mime="text/csv",
)
upload = st.file_uploader("Upload observation CSV", type=["csv"], key="pilot_csv_upload")
if upload is not None:
    try:
        accepted, rejected = parse_observation_csv(upload.getvalue())
    except (UnicodeError, ValueError) as exc:
        st.error(f"Import could not be processed: {exc}")
    else:
        eligible = [row for row in accepted if row["eligible_for_rssi_comparison"]]
        c1, c2, c3 = st.columns(3)
        c1.metric("Accepted records", len(accepted))
        c2.metric("RSSI comparison candidates", len(eligible))
        c3.metric("Rejected records", len(rejected))
        if accepted:
            st.dataframe(pd.DataFrame(accepted), use_container_width=True, hide_index=True)
            st.download_button(
                "⬇️ Export validated records",
                pd.DataFrame(accepted).to_csv(index=False),
                file_name="climatenetai_validated_observations.csv",
                mime="text/csv",
            )
        if rejected:
            st.warning("Some rows were rejected; correct the issues and upload again.")
            st.dataframe(pd.DataFrame(rejected), use_container_width=True, hide_index=True)
        st.caption(
            "Comparison candidates still require timestamp/location alignment "
            "with model predictions and provenance checks. Imported records are "
            "not automatically submitted to the RAC-5G telemetry monitor."
        )

st.divider()
st.subheader("Prospective RSSI comparison — research staging")
st.caption(
    "Optional: pair accepted RSSI measurements with independently recorded, "
    "timestamped predictions. This does not automatically update RAC-5G/RBE."
)
prediction_template = (
    "prediction_id,timestamp_utc,city,operator,predicted_rssi_dbm,"
    "lower_rssi_dbm,upper_rssi_dbm,provenance\\n"
    "illustrative-1,2026-10-13T13:00:00+00:00,Abuja,MTN,-82,-92,-72,"
    "illustrative_independent_prediction\\n"
).replace("\\n", "\n")
st.download_button(
    "⬇️ Download illustrative prediction-log format",
    prediction_template,
    file_name="climatenetai_prediction_log_example.csv",
    mime="text/csv",
)
pred_upload = st.file_uploader(
    "Upload previously recorded prediction log (CSV)",
    type=["csv"],
    key="pilot_prediction_log",
)
if upload is not None and pred_upload is not None:
    try:
        observations, observation_errors = parse_observation_csv(upload.getvalue())
        raw = pred_upload.getvalue()
        if len(raw) > 2_000_000:
            raise ValueError("Prediction log exceeds 2 MB")
        prediction_rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
        if len(prediction_rows) > 1000:
            raise ValueError("Prediction log exceeds 1000 rows")
        required_predictions = {
            "prediction_id", "timestamp_utc", "city", "operator",
            "predicted_rssi_dbm", "lower_rssi_dbm", "upper_rssi_dbm", "provenance",
        }
        if prediction_rows and not required_predictions.issubset(prediction_rows[0]):
            raise ValueError("Prediction log is missing required columns")
        if not prediction_rows:
            raise ValueError("Prediction log is empty")
        matches = []
        rejected_pairs = []
        used_observations = set()
        for prediction in prediction_rows:
            candidates = []
            for index, observation in enumerate(observations):
                if index in used_observations:
                    continue
                try:
                    match = pair_rssi_observation(observation, prediction)
                    candidates.append((match["Pairing_Gap_Minutes"], index, match))
                except (ValueError, KeyError):
                    continue
            if candidates:
                _, index, match = min(candidates, key=lambda entry: entry[0])
                used_observations.add(index)
                matches.append(match)
            else:
                rejected_pairs.append({
                    "prediction_id": prediction.get("prediction_id", ""),
                    "reason": "No eligible, prospective RSSI observation within pairing constraints",
                })
        st.metric("Candidate prospective pairs", len(matches))
        if matches:
            paired_df = pd.DataFrame(matches)
            st.dataframe(paired_df, use_container_width=True, hide_index=True)
            st.download_button(
                "⬇️ Download candidate comparison log",
                paired_df.to_csv(index=False),
                file_name="climatenetai_candidate_rssi_pairs.csv",
                mime="text/csv",
            )
        if rejected_pairs:
            with st.expander("Unmatched predictions"):
                st.dataframe(pd.DataFrame(rejected_pairs), hide_index=True, use_container_width=True)
        if observation_errors:
            st.warning(f"{len(observation_errors)} observation row(s) were invalid and excluded.")
        st.warning(
            "Candidate pairing is not external validation. Confirm device provenance, "
            "measurement location, radio metric definition, weather alignment, "
            "and prediction generation time before scientific evaluation. "
            "No results are added to RAC-5G reliability telemetry automatically."
        )
    except (ValueError, UnicodeError) as exc:
        st.error(f"Could not compare uploaded logs: {exc}")
