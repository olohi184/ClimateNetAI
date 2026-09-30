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

