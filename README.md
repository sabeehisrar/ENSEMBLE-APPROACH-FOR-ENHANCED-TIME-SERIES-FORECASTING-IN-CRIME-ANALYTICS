# Hybrid Time-Series Forecasting for Crime Analytics

A hybrid ensemble framework combining ARIMA, Artificial Neural Networks (ANN), and Random Forest to forecast monthly crime trends with enhanced accuracy and missing data resilience.

## Publication
This work was presented and published at the **2024 International Conference on Artificial Intelligence and Emerging Technology (Global AI Summit)**, Bennett University:

> **Title**: *A Synergistic Ensemble Approach for Enhanced Time Series Forecasting in Crime Analytics*  
> **Authors**: Reenu Rani, Rohan Mehta, Sabeeh Israr, Umar Tamboli, Rohith Murugan, Sanvi S. Chavan  
> **Publisher**: IEEE  
> **DOI / Link**: [10.1109/globalaisummit62156.2024.10947957](https://doi.org/10.1109/globalaisummit62156.2024.10947957)

---

## Problem
Accurate time-series forecasting in law enforcement is often impaired by missing data, non-linear trend fluctuations, and crime rate volatility. Standard statistical models struggle with non-linear patterns, while standalone neural networks are susceptible to overfitting on small or incomplete historical datasets. This project addresses these limitations by leveraging a multi-stage stacked ensemble architecture.

## Approach
- **Models/Techniques**: Auto-ARIMA baseline for linear temporal trends, Dense Artificial Neural Network (ANN) with Early Stopping for residual feature extraction, and Random Forest Meta-Regressor for stacked ensemble prediction.
- **Data**: Monthly Indian Penal Code (IPC) crime category time-series dataset.

## Results

| Task | Model | Accuracy / Metric |
| :--- | :--- | :--- |
| Crime Forecasting | ARIMA Baseline | Baseline MAPE |
| Crime Forecasting | Artificial Neural Network (ANN) | Non-Linear Residual MAPE |
| Crime Forecasting | **Hybrid ARIMA-ANN-Random Forest** | **+23% Accuracy Improvement vs individual models** |
| Data Quality Handling | Ensemble Residual Pipeline | **35% Reduced Impact of Missing Data** |

*Detailed visualization plots and metric breakdowns are available under [`results/`](results/results.md).*

## Tech Stack
- **Languages**: Python 3.11
- **Libraries & Frameworks**: `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `tensorflow` (Keras), `statsmodels`, `pmdarima`

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Execute analysis scripts in pipeline order under `/src`:
   ```bash
   python src/01_arima_baseline.py
   python src/02_ann_model.py
   python src/03_ensemble_arima_ann_rf.py
   ```

## Note on Data
The dataset used in this project was obtained via a formal legal request to Mumbai Police and is not publicly redistributable under the terms of that request. This repo contains the analysis methodology and code only. Contact the author for details on data access.
