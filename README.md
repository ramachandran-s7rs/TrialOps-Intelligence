# TrialOps Intelligence

TrialOps Intelligence is a Python-only Streamlit clinical data-quality and analytics platform for CDISC SDTM-style data. It supports session-scoped dataset uploads, structural domain detection, validation findings, persistent query management, subject review, safety/laboratory analytics, and treatment exposure and efficacy review.

## Run locally

```powershell
pip install -r requirements.txt
streamlit run app.py
```

## Supported detected domains

The application classifies data from standardized column signatures; it never uses filenames for business logic.

| Domain | Required structural signature |
| --- | --- |
| DM | `USUBJID`, `SEX`, `AGE` |
| AE | `USUBJID`, `AETERM` |
| LB | `USUBJID`, (`LBTEST` or `LBTESTCD`), (`LBSTRESN` or `LBORRES`) |
| VS | `USUBJID`, (`VSTEST` or `VSTESTCD`), (`VSSTRESN` or `VSORRES`) |
| EX | `USUBJID` plus an exposure variable such as `EXTRT`, `EXDOSE`, or `EXSTDTC` |
| EFF | `USUBJID`, (`PARAM` or `PARAMCD`), and at least one of `BASE`, `AVAL`, `CHG`, or `PCHG` |

## Clinical analytics

- **Data Upload** — upload one or more CSV datasets and retain detected datasets in the Streamlit session.
- **Validation Engine** — review SDTM data-quality findings and export results.
- **Query Management** — synchronize validation findings to SQLite-backed queries with audit history and lifecycle controls.
- **Dashboard** — review study, data-quality, safety, laboratory, treatment, and efficacy headline metrics.
- **Patient 360** — review demographics, exposure, efficacy, safety, laboratory, queries, and timeline data for one subject.
- **Laboratory Analytics** — classify results against supplied reference limits and review laboratory trends/alerts.
- **Efficacy Analytics** — review EX administration, arm assignments, endpoint summaries, change-from-baseline, treatment-arm safety denominators, and carefully labelled descriptive/inferential outputs.
- **Statistical Analysis** — conduct exploratory t-test or ANOVA treatment-arm review, effect-size, confidence-interval, responder, safety-versus-efficacy, and report-export analysis.
- **Anomaly Detection** — review explainable data-quality, laboratory, safety, visit-sequence, and site-operational signals with prioritization and exports.

`CHG` is derived as `AVAL - BASE` only when it is missing. `PCHG` is derived as `((AVAL - BASE) / BASE) * 100` only when it is missing and baseline is non-zero. The application does not make efficacy, safety, approval, or statistical-significance claims from descriptive output.

Statistical Analysis uses an independent Welch t-test for two evaluable arms and one-way ANOVA for more than two arms. It is an exploratory clinical-review utility, not a protocol-defined confirmatory analysis or FDA submission. PDF, Excel, and plain-text regulatory-review briefing downloads are generated locally from the active session.

## Synthetic demonstration data

The `data/generate_phase6_synthetic_data.py` script creates deterministic fictional Phase 6 demonstration CSVs for 60 subjects across Placebo, Drug A 50 mg, and Drug A 100 mg. Run it from the project root:

```powershell
python data/generate_phase6_synthetic_data.py
```

It creates `sample_phase6_dm.csv`, `sample_phase6_ex.csv`, `sample_phase6_ae.csv`, `sample_phase6_lb.csv`, and `sample_phase6_efficacy.csv` in `data/`. These files contain no real patient information.

## Project layout

```text
app.py                                # Navigation and Streamlit application entry point
database.py                           # SQLite queries, audit trail, and lab-alert persistence
pages/                                # Streamlit presentation pages
utils/domain_detection.py             # Filename-independent SDTM domain detection
utils/safety_analytics.py             # AE review helpers
utils/lab_analytics.py                # Laboratory classification and trends
utils/treatment_efficacy_analytics.py # Exposure, arm, efficacy, and arm-safety helpers
data/                                 # Local database and fictional demonstration datasets
```
