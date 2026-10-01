from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------

st.set_page_config(
    page_title="ClimateNetAI",
    page_icon="📡",
    layout="wide"
)


# ------------------------------------------------------------
# PROJECT PATHS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models" / "monthly_models"


# ------------------------------------------------------------
# CUSTOM STYLING
# ------------------------------------------------------------

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1400px;
        }

        h1, h2, h3 {
            letter-spacing: -0.02em;
        }

        .hero-card {
            padding: 1.4rem 1.6rem;
            border-radius: 18px;
            border: 1px solid rgba(128, 128, 128, 0.18);
            background: rgba(255, 255, 255, 0.86);
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.06);
            margin-bottom: 1.5rem;
        }

        .hero-title {
            font-size: 2.7rem;
            font-weight: 800;
            margin-bottom: 0.15rem;
        }

        .hero-subtitle {
            font-size: 1.05rem;
            color: #6b7280;
            margin-bottom: 0;
        }

        .section-title {
            font-size: 2rem;
            font-weight: 800;
            margin-top: 1.2rem;
            margin-bottom: 1rem;
        }

        .snapshot-card, .condition-card {
            border: 1px solid rgba(128, 128, 128, 0.16);
            border-radius: 16px;
            padding: 1.15rem 1.2rem;
            background: rgba(255, 255, 255, 0.9);
            box-shadow: 0 7px 20px rgba(15, 23, 42, 0.04);
            min-height: 118px;
        }

        .small-label {
            color: #6b7280;
            font-size: 0.95rem;
            margin-bottom: 0.35rem;
        }

        .big-value {
            font-size: 2rem;
            font-weight: 750;
            line-height: 1.1;
        }

        .prediction-panel {
            border: 1px solid rgba(128, 128, 128, 0.16);
            border-radius: 18px;
            padding: 1.35rem 1.4rem;
            background: rgba(250, 250, 250, 0.55);
            margin-top: 0.5rem;
        }

        div[data-testid="stButton"] > button {
            border-radius: 12px;
            min-height: 3rem;
            font-weight: 700;
        }

        div[data-testid="stDownloadButton"] > button {
            border-radius: 12px;
            min-height: 3rem;
            font-weight: 700;
        }

        section[data-testid="stSidebar"] {
            border-right: 1px solid rgba(128, 128, 128, 0.15);
        }

        .footer-note {
            color: #6b7280;
            font-size: 0.9rem;
            margin-top: 1rem;
        }
    
        /* ClimateNetAI immersive background */
        .stApp {
            background:
                radial-gradient(circle at 88% 8%, rgba(14,165,233,0.22), transparent 30rem),
                radial-gradient(circle at 12% 28%, rgba(16,185,129,0.13), transparent 27rem),
                radial-gradient(circle at 72% 78%, rgba(99,102,241,0.13), transparent 32rem),
                linear-gradient(135deg, #e8f5ff 0%, #eef7ff 38%, #eefcf8 70%, #edf2ff 100%) !important;
            background-attachment: fixed !important;
        }

        [data-testid="stAppViewContainer"] > .main {
            background: transparent !important;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #071a33 0%, #0b2a4a 52%, #0a3652 100%) !important;
            border-right: 1px solid rgba(125,211,252,0.22) !important;
        }

        [data-testid="stSidebar"] * {
            color: #eef8ff;
        }

        [data-testid="stSidebar"] input {
            color: #0f172a !important;
        }

        [data-testid="stSidebar"] div[data-baseweb="input"] {
            background: rgba(255,255,255,0.96) !important;
            border-radius: 12px !important;
        }

        [data-testid="stSidebar"] hr {
            border-color: rgba(255,255,255,0.16) !important;
        }

        /* Frosted research panels */
        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.84) !important;
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.78) !important;
            box-shadow: 0 10px 28px rgba(15,64,105,0.08) !important;
        }

        div[data-testid="stDataFrame"] {
            background: rgba(255,255,255,0.86) !important;
            box-shadow: 0 10px 30px rgba(15,64,105,0.07);
        }

        .workflow-step {
            background: rgba(255,255,255,0.86) !important;
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            border: 1px solid rgba(255,255,255,0.82) !important;
            box-shadow: 0 9px 24px rgba(15,64,105,0.08) !important;
        }

        /* Primary actions use ClimateNetAI blue rather than alarm red */
        div[data-testid="stButton"] button[kind="primary"] {
            background: linear-gradient(100deg, #0284c7 0%, #2563eb 55%, #4f46e5 100%) !important;
            border: 0 !important;
            color: white !important;
            box-shadow: 0 9px 22px rgba(37,99,235,0.22) !important;
        }

        /* Keep destructive/degradation alerts controlled by Streamlit alert semantics. */
        .block-container {
            padding-top: 1.15rem !important;
        }

    </style>
    """,
    unsafe_allow_html=True
)


# ------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

MONTH_NUMBERS = {
    month: index
    for index, month in enumerate(MONTHS, start=1)
}

MODEL_NAMES = {
    "Random Forest": "Random_Forest",
    "XGBoost": "XGBoost",
    "Decision Tree": "Decision_Tree",
    "Linear Regression": "Linear_Regression"
}


# ------------------------------------------------------------
# CLIMATENETAI V2 NAVIGATION / RAC-5G IMPLEMENTATION
# ------------------------------------------------------------

app_mode = st.sidebar.radio(
    "ClimateNetAI Module",
    ["RAC-5G Reliability Dashboard", "Legacy Monthly Model Explorer"],
    help=(
        "RAC-5G implements the reliability-aware framework developed from the "
        "252-observation field dataset. The legacy explorer preserves "
        "the original monthly-model interface for reproducibility."
    ),
)

if app_mode == "RAC-5G Reliability Dashboard":
    RAC_DATA_FILE = BASE_DIR / "rac5g_master_dataset_v1.csv"

    st.markdown(
        """
        <div class="hero-card" style="background:linear-gradient(120deg,#0f172a,#172554,#075985);color:white;box-shadow:0 18px 45px rgba(15,23,42,.16);border:none;">
            <div style="font-size:.78rem;font-weight:800;letter-spacing:.14em;text-transform:uppercase;color:#7dd3fc;margin-bottom:.55rem;">CLIMATE-AWARE NETWORK INTELLIGENCE</div>
            <div class="hero-title" style="color:white;">📡 ClimateNetAI <span style="color:#7dd3fc;">RAC-5G</span></div>
            <div class="hero-subtitle" style="color:#dbeafe;">
                Reliability-Aware Climate-Adaptive 5G Prediction Model
            </div>
            <div style="margin-top:1rem;display:flex;gap:.5rem;flex-wrap:wrap;font-size:.86rem;">
                <span style="padding:.38rem .65rem;border-radius:999px;background:rgba(255,255,255,.11);border:1px solid rgba(255,255,255,.16);">v2.0 Research Prototype</span>
                <span style="padding:.38rem .65rem;border-radius:999px;background:rgba(255,255,255,.11);border:1px solid rgba(255,255,255,.16);">252 Field Observations</span>
                <span style="padding:.38rem .65rem;border-radius:999px;background:rgba(255,255,255,.11);border:1px solid rgba(255,255,255,.16);">Abuja, Nigeria</span>
                <span style="padding:.38rem .65rem;border-radius:999px;background:rgba(255,255,255,.11);border:1px solid rgba(255,255,255,.16);">July 2024–June 2025</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="display:grid;grid-template-columns:repeat(8,1fr);gap:.45rem;margin:.8rem 0 1.15rem;">
          <div class="snapshot-card" style="min-height:0;padding:.65rem;text-align:center;">🌦️<br><b>Climate</b></div>
          <div class="snapshot-card" style="min-height:0;padding:.65rem;text-align:center;">📡<br><b>Predict</b></div>
          <div class="snapshot-card" style="min-height:0;padding:.65rem;text-align:center;">🎯<br><b>Uncertainty</b></div>
          <div class="snapshot-card" style="min-height:0;padding:.65rem;text-align:center;">📥<br><b>Observe</b></div>
          <div class="snapshot-card" style="min-height:0;padding:.65rem;text-align:center;">❤️<br><b>Monitor</b></div>
          <div class="snapshot-card" style="min-height:0;padding:.65rem;text-align:center;">⚠️<br><b>Detect</b></div>
          <div class="snapshot-card" style="min-height:0;padding:.65rem;text-align:center;">🔄<br><b>Adapt</b></div>
          <div class="snapshot-card" style="min-height:0;padding:.65rem;text-align:center;">✅<br><b>Verify</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "RAC-5G is a research prototype. It combines climate-aware RSSI prediction "
        "with calibrated uncertainty, telemetry feedback, model-health monitoring, "
        "degradation detection, controlled adaptation, and recovery verification."
    )

    if not RAC_DATA_FILE.exists():
        st.error(
            "RAC-5G dataset not found. Add `rac5g_master_dataset_v1.csv` to the "
            "repository root before running this module."
        )
        st.stop()

    @st.cache_data
    def load_rac_data(path):
        return pd.read_csv(path)

    @st.cache_resource
    def fit_rac_model(path):
        data = pd.read_csv(path)
        features = ["Temp_C", "Pressure_Clean_hPa", "RH_pct"]
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(data[features], data["RSSI_dBm"])
        return model

    rac_df = load_rac_data(RAC_DATA_FILE)
    rac_features = ["Temp_C", "Pressure_Clean_hPa", "RH_pct"]
    base_rac_model = fit_rac_model(RAC_DATA_FILE)

    # Frozen evidence from RAC-5G Experiments 7 and 11.
    NOMINAL_COVERAGE = 0.90
    FROZEN_INTERVAL_WIDTH_DB = 20.22
    FROZEN_HALF_WIDTH_DB = FROZEN_INTERVAL_WIDTH_DB / 2.0
    DEFAULT_WINDOW = 5
    DEFAULT_LAMBDA = 1.5
    DEFAULT_COVERAGE_FLOOR = 0.80

    # Reconstruct leakage-safe leave-one-month-out residuals for a historical
    # reference error distribution. This reproduces the frozen Ridge baseline.
    @st.cache_data
    def historical_oof_errors(path):
        data = pd.read_csv(path)
        residuals = []
        for held_month in data["Month_Label"].drop_duplicates():
            train = data[data["Month_Label"] != held_month]
            test = data[data["Month_Label"] == held_month]
            model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            model.fit(train[rac_features], train["RSSI_dBm"])
            pred = model.predict(test[rac_features])
            residuals.extend(np.abs(test["RSSI_dBm"].to_numpy() - pred))
        return np.asarray(residuals, dtype=float)

    historical_errors = historical_oof_errors(RAC_DATA_FILE)

    if "rac_model" not in st.session_state:
        st.session_state.rac_model = base_rac_model
    if "rac_telemetry" not in st.session_state:
        st.session_state.rac_telemetry = []
    if "rac_prediction" not in st.session_state:
        st.session_state.rac_prediction = None
    if "rac_adapted" not in st.session_state:
        st.session_state.rac_adapted = False
    if "rac_prediction_id" not in st.session_state:
        st.session_state.rac_prediction_id = 0
    if "rac_last_submitted_prediction_id" not in st.session_state:
        st.session_state.rac_last_submitted_prediction_id = None
    if "rac_adaptation_at" not in st.session_state:
        st.session_state.rac_adaptation_at = None
    if "rac_adaptation_completed" not in st.session_state:
        st.session_state.rac_adaptation_completed = False
    if "rac_recovery_verified" not in st.session_state:
        st.session_state.rac_recovery_verified = False

    st.sidebar.divider()
    st.sidebar.subheader("RAC-5G Climate Context")
    rac_temp = st.sidebar.number_input("Temperature (°C)", -20.0, 60.0, 30.0, 0.1, key="rac_temp")
    rac_pressure = st.sidebar.number_input("Pressure (hPa)", 800.0, 1200.0, 1000.0, 0.1, key="rac_pressure")
    rac_rh = st.sidebar.number_input("Relative Humidity (%)", 0.0, 100.0, 70.0, 0.1, key="rac_rh")
    predict_rac = st.sidebar.button("🔮 Generate RAC-5G Prediction", type="primary", width="stretch")

    if predict_rac:
        x_new = pd.DataFrame({
            "Temp_C": [rac_temp],
            "Pressure_Clean_hPa": [rac_pressure],
            "RH_pct": [rac_rh],
        })
        pred = float(st.session_state.rac_model.predict(x_new)[0])
        st.session_state.rac_prediction_id += 1
        st.session_state.rac_prediction = {
            "prediction_id": st.session_state.rac_prediction_id,
            "temperature": rac_temp,
            "pressure": rac_pressure,
            "humidity": rac_rh,
            "prediction": pred,
            "lower": pred - FROZEN_HALF_WIDTH_DB,
            "upper": pred + FROZEN_HALF_WIDTH_DB,
        }

    st.subheader("1. Prediction Intelligence — Climate → RSSI → Uncertainty")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Field observations", "252")
    k2.metric("Prediction Engine", "Ridge α = 1.0")
    k3.metric("Nominal Interval", "90%")
    k4.metric("Frozen Overall Coverage", "87.3%")

    current = st.session_state.rac_prediction
    if current is None:
        st.write(
            "Enter atmospheric conditions in the sidebar and generate a RAC-5G prediction. "
            "The uncertainty interval uses the frozen Experiment 7 interval width."
        )
    else:
        p1, p2, p3 = st.columns(3)
        p1.metric("Predicted RSSI", f"{current['prediction']:.2f} dBm")
        p2.metric("90% Lower Bound", f"{current['lower']:.2f} dBm")
        p3.metric("90% Upper Bound", f"{current['upper']:.2f} dBm")
        st.caption(
            "The frozen uncertainty experiment achieved 87.3% overall empirical coverage, "
            "but coverage varied strongly by month; the interval is therefore evidence, not a guarantee."
        )

        st.subheader("2. Live Telemetry Feedback")
        observed = st.number_input(
            "Observed RSSI after prediction (dBm)",
            min_value=-140.0,
            max_value=-30.0,
            value=float(round(current["prediction"], 1)),
            step=0.1,
            key="rac_observed_rssi",
        )
        already_submitted = (
            st.session_state.rac_last_submitted_prediction_id == current["prediction_id"]
        )
        if already_submitted:
            st.info("Telemetry for this prediction has already been submitted. Generate a new prediction before adding another observation.")
        elif st.button("📥 Submit Observed RSSI", width="stretch"):
            error = abs(float(observed) - current["prediction"])
            covered = int(current["lower"] <= float(observed) <= current["upper"])
            adaptation_active = st.session_state.rac_adaptation_completed or st.session_state.rac_adaptation_at is not None
            phase = "Post-adaptation" if adaptation_active else "Pre-adaptation"
            st.session_state.rac_telemetry.append({
                "Prediction_ID": current["prediction_id"],
                "Phase": phase,
                "Temperature_C": current["temperature"],
                "Pressure_hPa": current["pressure"],
                "Relative_Humidity_pct": current["humidity"],
                "Predicted_RSSI_dBm": current["prediction"],
                "Observed_RSSI_dBm": float(observed),
                "Absolute_Error_dB": error,
                "Interval_Covered": covered,
            })
            st.session_state.rac_last_submitted_prediction_id = current["prediction_id"]
            st.success("Telemetry observation added. Generate a new prediction before submitting the next observation.")

    st.subheader("3. Reliability Intelligence — Model Health & Degradation")
    d1, d2, d3 = st.columns(3)
    window = d1.selectbox("Monitoring window (w)", [3, 5, 7], index=1)
    lam = d2.selectbox("Threshold multiplier (λ)", [1.0, 1.5, 2.0, 2.5], index=1)
    coverage_floor = d3.slider("Operational coverage floor", 0.50, 1.00, DEFAULT_COVERAGE_FLOOR, 0.05)

    hist_median = float(np.median(historical_errors))
    hist_mad = float(np.median(np.abs(historical_errors - hist_median)))
    threshold = hist_median + float(lam) * (1.4826 * hist_mad)

    telemetry_df = pd.DataFrame(st.session_state.rac_telemetry)
    health = "Awaiting telemetry"
    rolling_mae = np.nan
    rolling_cov = np.nan

    if not telemetry_df.empty:
        recent = telemetry_df.tail(int(window))
        rolling_mae = float(recent["Absolute_Error_dB"].mean())
        rolling_cov = float(recent["Interval_Covered"].mean())
        enough = len(recent) >= int(window)
        error_fail = enough and rolling_mae > threshold
        coverage_fail = enough and rolling_cov < coverage_floor
        if error_fail and coverage_fail:
            health = "Degraded"
        elif error_fail or coverage_fail:
            health = "Warning"
        elif enough:
            health = "Stable"
        else:
            health = "Warming up"

    h1, h2, h3, h4 = st.columns(4)
    health_display = "Awaiting" if health == "Awaiting telemetry" else health
    h1.metric("State", health_display)
    h2.metric("Rolling MAE", "—" if np.isnan(rolling_mae) else f"{rolling_mae:.2f} dB")
    h3.metric("Rolling Coverage", "—" if np.isnan(rolling_cov) else f"{rolling_cov*100:.1f}%")
    h4.metric("Robust Error Threshold", f"{threshold:.2f} dB")

    st.caption(
        "The thesis sensitivity study tested w = 3–7 and λ = 1.0–2.5. "
        "The selected values are research settings, not universal telecom constants."
    )

    total_obs = len(telemetry_df)
    if st.session_state.rac_adaptation_at is None:
        post_obs = 0
    else:
        post_obs = max(0, total_obs - int(st.session_state.rac_adaptation_at))
    t1, t2, t3 = st.columns(3)
    t1.metric("Telemetry observations", total_obs)
    t2.metric("Post-adaptation observations", post_obs)
    adaptation_completed = (
        st.session_state.rac_adaptation_completed
        or st.session_state.rac_adaptation_at is not None
        or (not telemetry_df.empty and "Phase" in telemetry_df.columns and (telemetry_df["Phase"] == "Post-adaptation").any())
    )
    t3.metric("Current phase", "Post-adaptation" if adaptation_completed else "Pre-adaptation")

    if not telemetry_df.empty:
        st.dataframe(telemetry_df, use_container_width=True, hide_index=True)

    st.subheader("4. Adaptive Response — Controlled Adaptation & Recovery")

    # Verify recovery from POST-ADAPTATION telemetry only.
    post_mae = np.nan
    post_cov = np.nan
    post_health = "Awaiting post-adaptation telemetry"

    if adaptation_completed and post_obs > 0:
        post_df = telemetry_df.iloc[int(st.session_state.rac_adaptation_at):].copy()
        post_recent = post_df.tail(int(window))
        post_mae = float(post_recent["Absolute_Error_dB"].mean())
        post_cov = float(post_recent["Interval_Covered"].mean())

        if len(post_recent) < int(window):
            post_health = "Warming up"
        else:
            post_error_fail = post_mae > threshold
            post_coverage_fail = post_cov < coverage_floor
            if post_error_fail and post_coverage_fail:
                post_health = "Degraded"
            elif post_error_fail or post_coverage_fail:
                post_health = "Warning"
            else:
                post_health = "Stable"

    recovery_now = (
        adaptation_completed
        and post_obs >= int(window)
        and post_health == "Stable"
    )
    if recovery_now:
        st.session_state.rac_recovery_verified = True
    recovery_verified = st.session_state.rac_recovery_verified

    if health == "Degraded" and not adaptation_completed:
        st.error("RAC-5G has detected sustained reliability degradation.")
        if st.button("🔄 Trigger Controlled Adaptation", type="primary", width="stretch"):
            new_rows = telemetry_df.rename(columns={
                "Temperature_C": "Temp_C",
                "Pressure_hPa": "Pressure_Clean_hPa",
                "Relative_Humidity_pct": "RH_pct",
                "Observed_RSSI_dBm": "RSSI_dBm",
            })
            adapted_train = pd.concat([
                rac_df[rac_features + ["RSSI_dBm"]],
                new_rows[rac_features + ["RSSI_dBm"]],
            ], ignore_index=True)
            adapted_model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            adapted_model.fit(adapted_train[rac_features], adapted_train["RSSI_dBm"])
            st.session_state.rac_model = adapted_model
            st.session_state.rac_adapted = True
            st.session_state.rac_adaptation_completed = True
            st.session_state.rac_recovery_verified = False
            st.session_state.rac_adaptation_at = len(st.session_state.rac_telemetry)
            st.session_state.rac_prediction = None
            st.success(
                "Adaptation completed. Recovery status: PENDING. Generate new predictions and collect post-adaptation telemetry to verify recovery."
            )
    elif recovery_verified:
        st.success(
            f"Recovery status: VERIFIED. At least {int(window)} post-adaptation observations "
            f"have been collected. Post-adaptation MAE = {post_mae:.2f} dB "
            f"(threshold = {threshold:.2f} dB) and post-adaptation coverage = "
            f"{post_cov*100:.1f}% (floor = {coverage_floor*100:.1f}%)."
        )
    elif adaptation_completed and post_obs >= int(window) and post_health == "Degraded":
        st.error(
            f"Recovery status: NOT VERIFIED. The post-adaptation window remains Degraded "
            f"(MAE = {post_mae:.2f} dB; coverage = {post_cov*100:.1f}%). "
            "Continue monitoring or escalate for further model review."
        )
    elif adaptation_completed and post_obs >= int(window) and post_health == "Warning":
        st.warning(
            f"Recovery status: NOT VERIFIED. The post-adaptation window remains in Warning "
            f"(MAE = {post_mae:.2f} dB; coverage = {post_cov*100:.1f}%). "
            "Continue monitoring before declaring recovery."
        )
    elif adaptation_completed:
        st.warning(
            f"Recovery status: PENDING. {post_obs}/{int(window)} post-adaptation observations "
            "collected; adaptation alone is not proof of recovery."
        )
    else:
        st.info("Adaptation status: NOT REQUIRED. RAC-5G retains the current model while monitoring continues.")

    st.subheader("5. Validated Research Evidence")
    evidence = pd.DataFrame({
        "Evidence": [
            "Static Ridge baseline",
            "Conformal uncertainty",
            "Detected stress periods",
            "January post-alarm adaptation",
        ],
        "Frozen result": [
            "MAE 4.448 dB | RMSE 5.606 dB | R² −0.227",
            "90% nominal | 87.3% overall empirical coverage | 20.22 dB average full width",
            "November 2024, January 2025, June 2025",
            "MAE 21.74 → 7.42 dB (~65.9% reduction; case-specific)",
        ],
    })
    st.dataframe(evidence, use_container_width=True, hide_index=True)
    st.warning(
        "Do not interpret the January 65.9% reduction as a universal RAC-5G accuracy improvement. "
        "It is a case-specific recovery result from one severe degradation event."
    )

    if not telemetry_df.empty:
        st.download_button(
            "⬇️ Download RAC-5G Telemetry Log",
            data=telemetry_df.to_csv(index=False).encode("utf-8"),
            file_name="rac5g_telemetry_log.csv",
            mime="text/csv",
            width="stretch",
        )

    if st.button("🧹 Reset Validation Session", width="stretch"):
        st.session_state.rac_model = base_rac_model
        st.session_state.rac_telemetry = []
        st.session_state.rac_prediction = None
        st.session_state.rac_adapted = False
        st.session_state.rac_adaptation_completed = False
        st.session_state.rac_recovery_verified = False
        st.session_state.rac_prediction_id = 0
        st.session_state.rac_last_submitted_prediction_id = None
        st.session_state.rac_adaptation_at = None
        st.rerun()

    st.caption(
        "RAC-5G V1 implementation: Climate Context → RSSI Prediction → Calibrated Uncertainty → "
        "Telemetry Feedback → Reliability Monitoring → Degradation Detection → Controlled Adaptation → "
        "Recovery Verification."
    )
    st.stop()


# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------

st.sidebar.header("Prediction Settings")

month = st.sidebar.selectbox(
    "Select Month",
    MONTHS
)

restricted_linear_months = [
    "June",
    "July",
    "August",
    "October",
    "November",
    "December"
]

if month in restricted_linear_months:
    available_models = [
        model_name
        for model_name in MODEL_NAMES.keys()
        if model_name != "Linear Regression"
    ]
else:
    available_models = list(MODEL_NAMES.keys())

model_type = st.sidebar.selectbox(
    "Select Model",
    available_models
)

st.sidebar.divider()
st.sidebar.subheader("Environmental Inputs")

temperature = st.sidebar.number_input(
    "Temperature (°C)",
    min_value=-20.0,
    max_value=60.0,
    value=30.0,
    step=0.1
)

pressure = st.sidebar.number_input(
    "Pressure (hPa)",
    min_value=800.0,
    max_value=1200.0,
    value=1000.0,
    step=0.1
)

humidity = st.sidebar.number_input(
    "Relative Humidity (%)",
    min_value=0.0,
    max_value=100.0,
    value=70.0,
    step=0.1
)

month_number = MONTH_NUMBERS[month]

predict_button = st.sidebar.button(
    "🔮 Predict RSSI",
    type="primary",
    width="stretch"
)


# ------------------------------------------------------------
# MODEL FILE
# ------------------------------------------------------------

model_filename = f"{month}_{MODEL_NAMES[model_type]}.pkl"
model_path = MODEL_DIR / model_filename


# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">📡 ClimateNetAI</div>
        <div class="hero-subtitle">
            Climate-Aware Machine Learning for Wireless Signal Prediction
        </div>
        <div style="margin-top: 0.8rem; font-size: 0.92rem; color: #6b7280;">
           Research Prototype • Version 1.0.4
           cff-version: 1.2.0
message: "If you use ClimateNetAI in your research, please cite this software."
title: "ClimateNetAI: Climate-Aware Machine Learning for Wireless Signal Prediction"
version: 1.0.4
doi: "10.5281/zenodo.22731906"
date-released: "2026-09-13"
type: software
license: MIT
            &nbsp;&nbsp;|&nbsp;&nbsp;
            Developed by Olohimai Juliet Michael
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ------------------------------------------------------------
# RESEARCH SNAPSHOT
# ------------------------------------------------------------

st.markdown(
    '<div class="section-title">🔬 Research Snapshot</div>',
    unsafe_allow_html=True
)

s1, s2, s3, s4 = st.columns(4)

with s1:
    st.markdown(
        """
        <div class="snapshot-card">
            <div class="small-label">Dataset</div>
            <div class="big-value">221 records</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with s2:
    st.markdown(
        """
        <div class="snapshot-card">
            <div class="small-label">Environmental Variables</div>
            <div class="big-value">3</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with s3:
    st.markdown(
        """
        <div class="snapshot-card">
            <div class="small-label">ML Models</div>
            <div class="big-value">4</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with s4:
    st.markdown(
        """
        <div class="snapshot-card">
            <div class="small-label">Prediction Target</div>
            <div class="big-value">RSSI</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ------------------------------------------------------------
# CURRENT CONDITIONS
# ------------------------------------------------------------

st.markdown(
    '<div class="section-title">🌍 Current Prediction Conditions</div>',
    unsafe_allow_html=True
)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        f"""
        <div class="condition-card">
            <div class="small-label">Temperature</div>
            <div class="big-value">{temperature:.1f} °C</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c2:
    st.markdown(
        f"""
        <div class="condition-card">
            <div class="small-label">Pressure</div>
            <div class="big-value">{pressure:.1f} hPa</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c3:
    st.markdown(
        f"""
        <div class="condition-card">
            <div class="small-label">Humidity</div>
            <div class="big-value">{humidity:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c4:
    st.markdown(
        f"""
        <div class="condition-card">
            <div class="small-label">Month</div>
            <div class="big-value">{month}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ------------------------------------------------------------
# LIVE PREDICTION
# ------------------------------------------------------------

st.markdown(
    '<div class="section-title">🔮 Live RSSI Prediction</div>',
    unsafe_allow_html=True
)

st.write(
    "Enter environmental conditions in the sidebar and generate a prediction "
    "using one of the trained machine-learning models."
)

if not model_path.exists():
    st.warning(
        f"Model not found for {month} using {model_type}. "
        f"Expected file: {model_filename}"
    )
else:
    st.info(
        f"Ready to predict with **{model_type}** for **{month}**."
    )

if predict_button:

    if not model_path.exists():
        st.error(
            "Prediction cannot be generated because the selected model file is missing."
        )

    else:
        try:
            model = joblib.load(model_path)

            full_input = pd.DataFrame({
                "Temperature": [temperature],
                "Pressure": [pressure],
                "Relative_Humidity": [humidity],
                "Month_Number": [month_number]
            })

            # ------------------------------------------------
            # MODEL FEATURE COMPATIBILITY
            # ------------------------------------------------

            if hasattr(model, "feature_names_in_"):
                expected_features = list(model.feature_names_in_)

                missing_features = [
                    feature
                    for feature in expected_features
                    if feature not in full_input.columns
                ]

                if missing_features:
                    raise ValueError(
                        f"Model requires unavailable features: {missing_features}"
                    )

                model_input = full_input[expected_features]

            elif hasattr(model, "n_features_in_"):
                feature_count = int(model.n_features_in_)

                if feature_count == 4:
                    expected_features = [
                        "Temperature",
                        "Pressure",
                        "Relative_Humidity",
                        "Month_Number"
                    ]

                elif feature_count == 3:
                    expected_features = [
                        "Temperature",
                        "Pressure",
                        "Relative_Humidity"
                    ]

                else:
                    raise ValueError(
                        f"Unsupported model feature count: {feature_count}"
                    )

                model_input = full_input[expected_features]

            else:
                raise ValueError(
                    "Unable to determine the model's expected input features."
                )

            prediction = float(model.predict(model_input)[0])

            if prediction >= -85:
                quality = "Excellent"
            elif prediction >= -95:
                quality = "Good"
            elif prediction >= -105:
                quality = "Fair"
            else:
                quality = "Poor"

            st.success("Prediction completed successfully.")

            p1, p2 = st.columns([1, 1])

            with p1:
                st.metric(
                    "Predicted RSSI",
                    f"{prediction:.2f} dBm"
                )

            with p2:
                st.metric(
                    "Signal Quality",
                    quality
                )

            # ------------------------------------------------
            # PREDICTION INTERPRETATION
            # ------------------------------------------------

            st.subheader("🧭 Prediction Interpretation")

            if quality == "Excellent":
                interpretation = (
                    "The predicted RSSI indicates a strong signal condition. "
                    "Under these environmental inputs, the connection would generally "
                    "be expected to support stable wireless communication with a strong "
                    "received signal level."
                )

            elif quality == "Good":
                interpretation = (
                    "The predicted RSSI indicates a generally good signal condition. "
                    "Wireless connectivity should remain usable and relatively stable, "
                    "although performance may still vary with network load, interference, "
                    "location, and environmental conditions."
                )

            elif quality == "Fair":
                interpretation = (
                    "The predicted RSSI indicates a moderate signal condition. "
                    "Connectivity may remain usable, but reduced signal margin can make "
                    "performance more sensitive to atmospheric variability, interference, "
                    "mobility, and other propagation effects."
                )

            else:
                interpretation = (
                    "The predicted RSSI indicates a weak signal condition. "
                    "Connectivity may be degraded or unstable, and the link may be more "
                    "susceptible to environmental effects, interference, and coverage limitations."
                )

            st.info(interpretation)

            st.subheader("📊 Signal Assessment")

            a1, a2 = st.columns(2)

            with a1:
                st.info(
                    f"**Signal Quality:** {quality}"
                )

            with a2:
                st.info(
                    f"**Model:** {model_type}"
                )

            # ------------------------------------------------
            # MODEL RELIABILITY NOTICE
            # ------------------------------------------------

            if model_type == "Linear Regression":

                if month == "January":
                    st.success(
                        "✅ **Model Reliability:** January Linear Regression "
                        "was rebuilt, validated, and successfully deployed. "
                        "Its replacement model showed excellent validation performance."
                    )

                elif month == "May":
                    st.warning(
                        "⚠️ **Model Reliability Notice:** The May Linear Regression "
                        "model produces plausible predictions, but its out-of-sample "
                        "validation performance was modest. Interpret this monthly "
                        "prediction with caution."
                    )

                elif month in ["June", "July"]:
                    st.warning(
                        "⚠️ **Data Quality Notice:** June and July contain identical "
                        "Temperature, Pressure, Relative Humidity, and RSSI observations "
                        "in the current modeling dataset. Predictions for these months "
                        "should not be treated as independent monthly evidence until "
                        "the original source data is verified."
                    )

                elif month in ["August", "October", "November", "December"]:
                    st.warning(
                        "⚠️ **Model Reliability Notice:** This monthly Linear Regression "
                        "model showed limited out-of-sample predictive performance during "
                        "validation. The RSSI prediction may be plausible, but it should "
                        "be interpreted cautiously and alongside the model validation results."
                    )

            # ------------------------------------------------
            # MODEL VALIDATION PERFORMANCE
            # ------------------------------------------------

            performance_file = BASE_DIR / "monthly_model_results.csv"

            if performance_file.exists():
                performance_df = pd.read_csv(performance_file)

                performance_df["Month"] = performance_df["Month"].astype(str).str.strip()
                performance_df["Model"] = performance_df["Model"].astype(str).str.strip()

                selected_performance = performance_df[
                    (performance_df["Month"] == month)
                    & (performance_df["Model"] == model_type)
                ]

                if not selected_performance.empty:
                    performance_row = selected_performance.iloc[0]

                    mae_value = float(performance_row["MAE"])
                    rmse_value = float(performance_row["RMSE"])
                    r2_value = float(performance_row["R2"])

                    st.subheader("📈 Model Validation Performance")

                    m1, m2, m3 = st.columns(3)

                    with m1:
                        st.metric("MAE", f"{mae_value:.4f}")

                    with m2:
                        st.metric("RMSE", f"{rmse_value:.4f}")

                    with m3:
                        st.metric("R²", f"{r2_value:.4f}")

                    if r2_value >= 0.75:
                        performance_label = "Strong validation performance"
                    elif r2_value >= 0.50:
                        performance_label = "Moderate validation performance"
                    elif r2_value >= 0:
                        performance_label = "Limited validation performance"
                    else:
                        performance_label = "Poor out-of-sample generalization"

                    st.caption(
                        f"Validation assessment: {performance_label}"
                    )

                else:
                    st.info(
                        "Validation metrics are not available "
                        "for this month/model combination."
                    )

            # ------------------------------------------------
            # BEST MODEL FOR SELECTED MONTH
            # ------------------------------------------------

            best_model_file = BASE_DIR / "best_model_per_month_validated.csv"

            if best_model_file.exists():
                best_model_df = pd.read_csv(best_model_file)

                best_model_df["Month"] = best_model_df["Month"].astype(str).str.strip()
                best_model_df["Model"] = best_model_df["Model"].astype(str).str.strip()

                month_best = best_model_df[
                    best_model_df["Month"] == month
                ]

                if not month_best.empty:
                    best_row = month_best.iloc[0]

                    best_model_name = str(best_row["Model"])
                    best_mae = float(best_row["MAE"])
                    best_rmse = float(best_row["RMSE"])
                    best_r2 = float(best_row["R2"])
                    recommendation_status = str(best_row["Recommendation_Status"])

                    st.subheader("🏆 Best Model for Selected Month")

                    if recommendation_status == "Recommended":
                        st.success(
                            f"**Recommended model for {month}: {best_model_name}**  \n"
                            f"R² = {best_r2:.4f} | RMSE = {best_rmse:.4f} | MAE = {best_mae:.4f}"
                        )

                    else:
                        st.warning(
                            f"**Highest-ranked model for {month}: {best_model_name}**  \n"
                            f"R² = {best_r2:.4f} | RMSE = {best_rmse:.4f} | MAE = {best_mae:.4f}  \n"
                            "No model for this month demonstrated positive out-of-sample R²."
                        )

            # ------------------------------------------------
            # MODEL COMPARISON FOR SELECTED MONTH
            # ------------------------------------------------

            st.subheader("📊 Model Comparison for Selected Month")

            month_comparison = performance_df[
                performance_df["Month"] == month
            ][
                ["Model", "MAE", "RMSE", "R2"]
            ].copy()

            if not month_comparison.empty:

                month_comparison = month_comparison.sort_values(
                    by="R2",
                    ascending=False
                )

                month_comparison["MAE"] = month_comparison["MAE"].round(4)
                month_comparison["RMSE"] = month_comparison["RMSE"].round(4)
                month_comparison["R2"] = month_comparison["R2"].round(4)

                month_comparison = month_comparison.rename(
                    columns={
                        "R2": "R²"
                    }
                )

                st.dataframe(
                    month_comparison,
                    use_container_width=True,
                    hide_index=True
                )

                st.caption(
                    "Models are ordered from highest to lowest validation R². "
                    "MAE and RMSE are also shown for comparison."
                )

            else:
                st.info(
                    "Model comparison results are not available "
                    "for the selected month."
                )

            st.subheader("🌍 Environmental Conditions Used")

            result_table = pd.DataFrame({
                "Parameter": [
                    "Temperature",
                    "Pressure",
                    "Relative Humidity",
                    "Month"
                ],
                "Value": [
                    f"{temperature:.2f} °C",
                    f"{pressure:.2f} hPa",
                    f"{humidity:.2f} %",
                    month
                ]
            })

            st.dataframe(
                result_table,
                width="stretch",
                hide_index=True
            )

            with st.expander("Technical model details"):
                st.write(
                    f"**Model file:** `{model_filename}`"
                )
                st.write(
                    f"**Features used:** {', '.join(expected_features)}"
                )
                st.write(
                    f"**Feature count:** {len(expected_features)}"
                )

            report_df = pd.DataFrame({
                "Month": [month],
                "Model": [model_type],
                "Temperature_C": [temperature],
                "Pressure_hPa": [pressure],
                "Relative_Humidity_pct": [humidity],
                "Month_Number": [month_number],
                "Predicted_RSSI_dBm": [prediction],
                "Signal_Quality": [quality],
                "Features_Used": [", ".join(expected_features)],
                "Validation_MAE": [mae_value],
                "Validation_RMSE": [rmse_value],
                "Validation_R2": [r2_value],
                "Validation_Assessment": [performance_label],
                "Monthly_Best_Model": [best_model_name],
                "Monthly_Best_MAE": [best_mae],
                "Monthly_Best_RMSE": [best_rmse],
                "Monthly_Best_R2": [best_r2],
                "Recommendation_Status": [recommendation_status]
            })

            st.download_button(
                "⬇️ Download Prediction Report",
                data=report_df.to_csv(index=False).encode("utf-8"),
                file_name="climatenetai_prediction.csv",
                mime="text/csv",
                width="stretch"
            )

        except Exception as error:
            st.error(
                "An error occurred while generating the prediction."
            )
            st.exception(error)


# ------------------------------------------------------------
# MODEL AVAILABILITY
# ------------------------------------------------------------

st.divider()

with st.expander("🗂️ View Monthly Model Availability"):
    availability_rows = []

    for current_month in MONTHS:
        row = {"Month": current_month}

        for display_name, file_name in MODEL_NAMES.items():
            path = MODEL_DIR / f"{current_month}_{file_name}.pkl"
            row[display_name] = (
                "✅ Available"
                if path.exists()
                else "⚠️ Missing"
            )

        availability_rows.append(row)

    availability_df = pd.DataFrame(availability_rows)

    st.dataframe(
        availability_df,
        width="stretch",
        hide_index=True
    )


# ------------------------------------------------------------
# ABOUT
# ------------------------------------------------------------

st.divider()
# ------------------------------------------------------------
# MONTHLY MODEL PERFORMANCE TRENDS
# ------------------------------------------------------------

st.subheader("📈 Monthly Model Performance Trends")

trend_file = BASE_DIR / "monthly_model_results.csv"

if trend_file.exists():
    trend_df = pd.read_csv(trend_file)

    trend_df["Month"] = trend_df["Month"].astype(str).str.strip()
    trend_df["Model"] = trend_df["Model"].astype(str).str.strip()

    month_order = [
        "January", "February", "March", "April",
        "May", "June", "July", "August",
        "September", "October", "November", "December"
    ]

    trend_df["Month"] = pd.Categorical(
        trend_df["Month"],
        categories=month_order,
        ordered=True
    )

    r2_chart = trend_df.pivot(
        index="Month",
        columns="Model",
        values="R2"
    ).reindex(month_order)

    st.line_chart(
        r2_chart,
        use_container_width=True
    )

    st.caption(
        "Monthly out-of-sample R² performance for the four machine-learning models. "
        "Higher values indicate stronger predictive performance; negative R² values "
        "indicate poor generalization relative to predicting the test-set mean."
    )

    st.info(
        "June and July should be interpreted cautiously because the current modeling "
        "dataset contains identical environmental and RSSI observations for both months."
    )

else:
    st.info(
        "Monthly performance trend data are currently unavailable."
    )

st.divider()

# ------------------------------------------------------------
# RESEARCH AND DATA QUALITY
# ------------------------------------------------------------

st.subheader("🔬 Research & Data Quality")

with st.expander(
    "View validation and data-quality notes",
    expanded=False
):

    st.markdown("**Model validation**")
    st.write(
        "ClimateNetAI reports MAE, RMSE, and R² to provide transparent "
        "information about the predictive performance of each monthly model. "
        "Model recommendations are based primarily on validation R² and should "
        "be interpreted together with the reported error metrics."
    )

    st.markdown("**June and July data-quality notice**")
    st.warning(
        "The current modeling dataset contains identical Temperature, Pressure, "
        "Relative Humidity, and RSSI observations for June and July. Results for "
        "these two months should therefore not be treated as independent monthly "
        "evidence until the original source data has been verified."
    )

    st.markdown("**September validation note**")
    st.info(
        "September contains 8 observations in the current modeling dataset. "
        "Its MAE, RMSE, and R² values were reconstructed using Leave-One-Out "
        "Cross Validation (LOOCV) with Temperature, Pressure, Relative Humidity, "
        "and Month Number as model features."
    )

    st.markdown("**Interpretation of negative R²**")
    st.write(
        "A negative validation R² does not mean that the application failed to "
        "generate a prediction. It indicates that the model generalized poorly "
        "on the validation observations relative to a simple mean-prediction "
        "baseline. ClimateNetAI therefore avoids presenting such models as "
        "recommended models."
    )

    st.caption(
        "These notes are included to support transparent interpretation of "
        "ClimateNetAI outputs and should be considered when using results for "
        "research, reporting, or decision support."
    )

st.divider()

st.subheader("ℹ️ About ClimateNetAI")

st.write(
    """
ClimateNetAI is a climate-aware machine-learning research application
developed to demonstrate RSSI prediction from real-world atmospheric
measurements.

The deployment supports both legacy 3-feature monthly models
(Temperature, Pressure and Relative Humidity) and newer 4-feature models
that additionally use Month Number. The saved trained models are not
altered by the dashboard.
"""
)

st.markdown(
    '<div class="footer-note">Research demonstration — interpret predictions together with model-validation results and study limitations.</div>',
    unsafe_allow_html=True
)

