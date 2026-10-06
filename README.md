# ClimateNetAI

## Reliability-Aware Climate-Adaptive 5G Intelligence

![Version](https://img.shields.io/badge/version-2.3-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/status-research%20prototype-orange)

**Founder, Project Originator and Lead Developer:** Olohimai Juliet Michael  
**Current Version:** 2.3  
**Status:** Research Prototype

---

## Live Application

ClimateNetAI is publicly accessible at:

https://climatenetai.streamlit.app

ClimateNetAI is a climate-aware and reliability-aware machine-learning
research platform for investigating and predicting 5G Received Signal
Strength Indicator (RSSI) under atmospheric variability.

Version 2.2 implements the **RAC-5G (Reliability-Aware Climate-Adaptive
5G) framework**, extending static signal prediction into a workflow that
combines climate context, uncertainty quantification, network observation,
reliability monitoring, degradation detection, controlled adaptation,
and post-adaptation recovery verification.

---

## ClimateNetAI v2.3

ClimateNetAI v2.3 represents a research-to-software translation of
doctoral research investigating machine-learning prediction of 5G signal
behaviour under atmospheric variability.

The current system follows the operational workflow:

**Climate → Predict → Quantify Uncertainty → Observe → Monitor → Detect → Adapt → Verify**

The platform currently provides:

- climate-aware 5G RSSI prediction;
- live atmospheric-data integration;
- manual atmospheric-input fallback;
- prediction uncertainty intervals;
- a network telemetry bridge;
- rolling prediction-error monitoring;
- empirical interval-coverage monitoring;
- reliability-state classification;
- sustained degradation detection;
- controlled model adaptation; and
- post-adaptation recovery verification.

---

## RAC-5G Framework

**RAC-5G** stands for:

**Reliability-Aware Climate-Adaptive 5G Prediction Model**

RAC-5G is the reliability-aware framework implemented within ClimateNetAI.

The current V1 prediction engine uses a regularized linear model with
atmospheric predictors:

- Temperature
- Atmospheric Pressure
- Relative Humidity

The prediction target is:

- Received Signal Strength Indicator (RSSI)

RAC-5G is not presented as a new regression algorithm. Its contribution
is the integration of prediction, uncertainty, reliability monitoring,
degradation detection, controlled adaptation, and recovery verification
within a climate-aware 5G research workflow.

---

## Empirical Research Foundation

The RAC-5G implementation is grounded in an empirical 5G and atmospheric
field experiment conducted in Abuja, Nigeria.

The primary study dataset covers:

**July 2024 – June 2025**

and contains:

**252 field observations**

The research dataset contains atmospheric variables and wireless-network
measurements collected during the study period.

Core variables used by the current RAC-5G prediction engine are:

- Temperature (°C)
- Atmospheric Pressure (hPa)
- Relative Humidity (%)
- RSSI (dBm)

Additional network measurements collected during the broader research
include signal and network-performance variables such as RSRP, ping,
download speed, and upload speed.

The empirical field dataset and the telemetry used for software
functional validation should not be interpreted as the same source of
evidence.

---

## Live Climate API Integration

ClimateNetAI supports live atmospheric context through integration with
the **Open-Meteo API**.

The live climate pathway retrieves current:

- Temperature
- Relative Humidity
- Surface Pressure

These variables can be supplied to the RAC-5G prediction engine as the
environmental context for generating a current RSSI prediction.

Conceptually:

**Open-Meteo → ClimateNetAI → RAC-5G Prediction**

ClimateNetAI also retains a **Manual Climate Input** mode.

Manual mode provides an explicit fallback when external API or network
services are unavailable and also supports controlled research testing.

The system does not intentionally replace unavailable API observations
with hidden synthetic climate values.

---

## Prediction Uncertainty

ClimateNetAI does not present RSSI predictions solely as point estimates.

The RAC-5G interface provides:

- Predicted RSSI
- Lower uncertainty bound
- Upper uncertainty bound

The current V1 implementation uses a frozen empirical uncertainty interval
derived from the research validation workflow.

The nominal interval coverage is:

**90%**

with a frozen full interval width of approximately:

**20.22 dB**

The uncertainty interval is intended to communicate predictive uncertainty
and support subsequent reliability monitoring.

---

## Network Telemetry Bridge

ClimateNetAI v2.3 retains the **Telemetry Bridge** that allows network
observations to be supplied independently to the application.

Conceptually:

**Network Measurement Source → Telemetry Bridge → ClimateNetAI → RAC-5G Reliability Monitoring**

The bridge can provide:

- Observed RSSI
- Operator
- Network type
- Measurement timestamp
- Optional RSRP
- Optional RSRQ
- Optional SINR
- Source/provenance information

The current RAC-5G V1 reliability workflow evaluates **RSSI** as the
primary observed prediction target.

Additional radio measurements are retained as contextual telemetry and
provide a pathway for future extensions.

---

## Reliability Monitoring

After a prediction has been generated, an observed RSSI measurement can
be compared with the predicted value.

ClimateNetAI monitors recent prediction performance using a rolling
window.

The principal reliability indicators are:

- Rolling Mean Absolute Error (MAE)
- Empirical Prediction-Interval Coverage

The system classifies operational reliability states such as:

- Warming up
- Stable
- Warning
- Degraded

A reliability threshold is derived from historical prediction-error
behaviour using a robust median/MAD-based formulation.

The current default rolling monitoring window is:

**5 observations**

---

## Reliability Boundary Engine (RBE)

ClimateNetAI v2.3 adds a **Reliability Boundary Engine (RBE)** as a
selective-prediction layer above the existing RAC-5G telemetry monitor.

RBE does not replace the RAC-5G **Stable / Warning / Degraded** model-health
states. Those states continue to support degradation detection, adaptation,
and recovery. RBE answers a different operational question: whether the
**current prediction** is sufficiently supported by recent completed
telemetry to be presented as **TRUST**, **CAUTION**, or **ABSTAIN**.

The validated research configuration is frozen at:

- monitoring window: **5 completed observations**;
- robust error threshold: **8.71 dB**;
- empirical coverage floor: **0.80**; and
- conformal prediction-interval half-width: **10.11 dB**.

The first five observations form a warm-up period. For every subsequent
prediction, RBE computes its decision from the **preceding five completed
telemetry observations only**. The current observed RSSI is not available
to the RBE when that prediction is classified, preventing outcome leakage.

Decision rule:

- **TRUST** — neither recent-error nor recent-coverage criterion fails;
- **CAUTION** — exactly one criterion fails;
- **ABSTAIN** — both criteria fail.

In retrospective temporal validation, RBE states separated prediction risk:
TRUST predictions had lower error and higher interval coverage than ABSTAIN
predictions. This is evidence of risk stratification within the study data,
not a guarantee that every TRUST prediction will be accurate.

## Degradation Detection

RAC-5G does not trigger adaptation simply because a new observation
arrives.

Instead, ClimateNetAI evaluates whether recent prediction behaviour
indicates sustained reliability degradation.

This design separates:

**observation**

from

**adaptation**

and allows model updates to be triggered by evidence of degraded
reliability rather than by continuous retraining.

---

## Controlled Adaptation

When the reliability state satisfies the degradation criterion,
ClimateNetAI can enable controlled adaptation.

The adaptation process incorporates available telemetry observations
into the model-update workflow while retaining the empirical historical
training data.

Adaptation is therefore treated as a controlled response to detected
reliability degradation.

Importantly, completing adaptation does **not** automatically mean that
the system has recovered.

---

## Recovery Verification

Following adaptation, ClimateNetAI requires new post-adaptation
observations.

Recovery is evaluated only using the post-adaptation monitoring period.

The system requires sufficient post-adaptation evidence before declaring
recovery.

This implements the principle:

**Adaptation ≠ Recovery**

Instead:

**Adaptation → New Observations → Reliability Re-evaluation → Recovery Verification**

---

## Controlled Functional Validation of v2.2

The ClimateNetAI v2.2 telemetry and reliability workflow was functionally
tested using controlled simulated 5G observations supplied through the
Telemetry Bridge.

The validation sequence exercised:

**Stable Operation → Induced Degradation → Controlled Adaptation → Post-Adaptation Recovery**

During the stable phase, five controlled observations produced:

- Rolling MAE: approximately **0.40 dB**
- Rolling interval coverage: **100%**
- Reliability state: **Stable**

A sequence of deliberately degraded RSSI observations was subsequently
introduced to test degradation detection.

ClimateNetAI detected sustained reliability degradation and enabled the
controlled-adaptation pathway.

Adaptation was triggered once.

The system did not immediately declare recovery.

Five new post-adaptation observations were then supplied.

The final post-adaptation evaluation produced:

- Post-adaptation observations: **5**
- Post-adaptation MAE: **0.23 dB**
- Post-adaptation interval coverage: **100%**
- Reliability state: **Stable**
- Recovery status: **VERIFIED**

These values describe the controlled software-validation scenario and
should not be interpreted as independent field-performance estimates.

---

## Important Validation Distinction

Two forms of evidence should be distinguished when interpreting
ClimateNetAI v2.2.

### 1. Empirical Research Evidence

The underlying research and RAC-5G development are grounded in actual
5G and atmospheric field measurements collected during the twelve-month
Abuja experiment.

### 2. Software Functional Validation

The v2.2 Telemetry Bridge was functionally validated using controlled
simulated 5G observations.

The simulator is a software test harness.

It is **not** a substitute for independent real-time 5G field telemetry
and is not presented as evidence of direct integration with MTN, Airtel,
or another telecommunications operator.

Direct real-time integration with compatible 5G measurement devices or
operator telemetry remains future work.

---

## System Architecture

The current ClimateNetAI architecture can be summarized as:

```text
                  LIVE CLIMATE PATHWAY

              Open-Meteo Climate API
                       |
                       v
              Atmospheric Context
          Temperature / Pressure / RH
                       |
                       v
                 +-------------+
                 | ClimateNetAI|
                 +-------------+
                       |
                       v
                    RAC-5G
                       |
              +--------+--------+
              |                 |
              v                 v
       RSSI Prediction     Uncertainty
              |                 |
              +--------+--------+
                       |
                       v
              Reliability Engine
                       ^
                       |
              Telemetry Bridge
                       ^
                       |
              Network Observation

                       |
                       v
       Error + Coverage Monitoring
                       |
                       v
            Degradation Detection
                       |
                       v
          Controlled Adaptation
                       |
                       v
          Recovery Verification
