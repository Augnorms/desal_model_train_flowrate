# 🧪 RO Desalination Machine Learning Model Training

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)]()
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Random%20Forest-F7931E?logo=scikitlearn&logoColor=white)]()
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)]()
[![Status](https://img.shields.io/badge/Status-Trained%20%26%20Validated-brightgreen.svg)]()

This repository contains the machine learning training pipeline for predicting **Reverse Osmosis (RO) Desalination Plant** performance metrics. It trains dual **Random Forest Regressors** using historical and simulated operational parameters to predict product water quantity and quality.

---

## 🎯 Objectives

The models predict two primary performance indicators of the desalination process:

1. **Permeate Flowrate ($m^3/h$)** — Measures product water production rate (quantity).
2. **Permeate TDS ($ppm$)** — Measures product water Total Dissolved Solids / salinity (purity & quality).

---

## 📊 Dataset & Features

The model is trained on `clean_patterned_desalination_dataset.xlsx`.

### Input Features (Predictors)
| Feature | Units | Description |
| :--- | :--- | :--- |
| **Feed Flowrate** | $m^3/h$ | Rate of raw seawater/brackish water fed into the membranes |
| **Feed TDS** | $ppm$ | Total dissolved solids in the feed water |
| **Feed Salinity** | $g/L$ | Salt concentration in the feed water |
| **Feed Temperature** | $°C$ | Water temperature entering the membrane modules |
| **Recovery** | $\%$ | Percentage of feed water converted to permeate water |
| **Flux** | $LMH$ ($L/m^2/h$) | Volumetric permeate flux through the membrane area |
| **Pressure Drop** | $bar$ | Hydraulic pressure drop across the membrane elements |

### Target Variables (Outputs)
| Target Variable | Units | Description |
| :--- | :--- | :--- |
| **Permeate Flowrate** | $m^3/h$ | Predicts fresh water volume produced |
| **Permeate TDS** | $ppm$ | Predicts water purity (<50: Excellent, <100: Good, <200: Warning) |

---

## 🧠 Modeling Methodology

- **Algorithm**: `RandomForestRegressor(n_estimators=100, random_state=42)`
- **Data Splitting**: 80% Training / 20% Testing split (`random_state=42`)
- **Evaluation Metrics**:
  - **$R^2$ Score**: Proportion of variance explained by the model (max 1.0)
  - **Mean Absolute Error (MAE)**: Average magnitude of prediction errors in actual units
- **Feature Importance**: Analyzes and ranks which operational parameters impact flowrate and TDS most heavily.

---

## 📈 Model Performance & Benchmark Results

The models were trained on 16,000 records (80%) and evaluated against an unseen holdout test set of 4,000 records (20%, `random_state=42`):

| Target Indicator | Model Algorithm | $R^2$ Score | Mean Absolute Error (MAE) | Primary Feature Driver |
| :--- | :--- | :---: | :---: | :--- |
| **Permeate Flowrate** ($m^3/h$) | Random Forest Regressor | **0.9848** | **0.80 $m^3/h$** | Flux (75.4%) |
| **Permeate TDS** ($ppm$) | Random Forest Regressor | **0.9599** | **9.49 $ppm$** | Feed TDS (79.3%) |

### 🔍 Top Influential Features

- **For Permeate Flowrate**:
  1. `Flux (LMH)` — 75.38%
  2. `Feed Flowrate (m³/h)` — 10.87%
  3. `Recovery (%)` — 8.76%
  4. `Feed Temperature (°C)` — 3.36%

- **For Permeate TDS**:
  1. `Feed TDS (ppm)` — 79.26%
  2. `Feed Salinity (g/L)` — 10.38%
  3. `Feed Temperature (°C)` — 6.59%
  4. `Pressure Drop (bar)` — 1.46%


## 📁 Repository Structure

```text
train_model_desal/
├── models/
│   └── saved_models/
│       ├── permeate_flowrate_model_rf.pkl   # Serialized flowrate model
│       └── permeate_flowtds_model_rf.pkl    # Serialized TDS model
├── clean_patterned_desalination_dataset.xlsx # Cleaned training dataset
├── desalination_records.xlsx                 # Historical records
├── train_model.py                            # Model training & evaluation pipeline
├── requirements.txt                          # Python dependencies
├── .gitignore
└── README.md
