"""Research pilot: review independently submitted mobile measurements.

This page intentionally does not claim direct Android acquisition or
validated live RSSI accuracy. The existing RAC-5G prediction/telemetry
workflow remains in app.py; this page is an ingestion and QA staging area.
"""
import pandas as pd
import streamlit as st

from src.live_csv import parse_observation_csv

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
