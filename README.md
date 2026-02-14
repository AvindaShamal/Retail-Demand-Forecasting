# 📦 End-to-End MLOps Pipeline for Retail Demand Forecasting (AWS)

## 🔍 Project Overview

This project implements an **industry-grade MLOps pipeline** for **retail demand forecasting** using the Walmart Sales dataset.  
The system is designed to reflect **real-world production ML workflows**, including:

- Time-series feature engineering
- Model training with XGBoost
- Experiment tracking and model registry using MLflow
- Batch inference for demand forecasting
- CI/CD for deployment
- Cloud-native deployment on AWS
- Monitoring and basic drift detection

The project intentionally uses **batch prediction**, as demand forecasting is not latency-sensitive and aligns with real retail industry practices.

---

## 🎯 Business Problem

Retailers must forecast product demand accurately to:
- Avoid overstocking and stockouts
- Optimize inventory planning
- Improve revenue and supply chain efficiency

Demand patterns change over time due to:
- Seasonality
- Holidays
- Economic conditions

This project addresses these challenges by building a **continuously trainable and deployable forecasting system**.

---

## 📊 Dataset

**Walmart Store Sales Dataset (Kaggle)**

- Weekly sales data
- Multiple stores and departments
- External features:
  - Holidays
  - Fuel price
  - CPI
  - Unemployment rate

📌 The dataset enables realistic simulation of **concept drift** and **model retraining**.

---

## 🧠 ML Problem Definition

- **Task:** Time-series regression
- **Target:** Weekly sales
- **Forecast Horizon:** Configurable (e.g., next 4 weeks)
- **Evaluation Metrics:** RMSE, MAE
- **Baseline Model:** XGBoost Regressor

---

## 🏗️ System Architecture

        ┌──────────────┐
        │ Walmart Data │
        │   (CSV)      │
        └──────┬───────┘
               │
        ┌──────▼───────┐
        │   Amazon S3  │  ← Raw & processed data
        └──────┬───────┘
               │
     ┌─────────▼─────────┐
     │ Training Pipeline │
     │ (EC2 / ECS)       │
     └─────────┬─────────┘
               │
     ┌─────────▼─────────┐
     │     MLflow        │  ← Experiments & Model Registry
     │ (EC2 + S3)        │
     └─────────┬─────────┘
               │
     ┌─────────▼─────────┐
     │ Batch Inference   │
     │ (Scheduled Job)  │
     └─────────┬─────────┘
               │
     ┌─────────▼─────────┐
     │ Predictions Store │
     │ (RDS / S3)        │
     └─────────┬─────────┘
               │
     ┌─────────▼─────────┐
     │ FastAPI Backend   │
     │ (ECS / EC2)       │
     └─────────┬─────────┘
               │
        ┌──────▼───────┐
        │ End User /   │
        │ Dashboard    │
        └──────────────┘


### High-Level Components

- **Amazon S3**
  - Raw data storage
  - Processed feature data
  - Model artifacts
  - Batch predictions

- **Training & Inference**
  - EC2 / ECS (CPU-based)
  - XGBoost model training
  - Scheduled batch inference jobs

- **Experiment Tracking**
  - MLflow (hosted on EC2)
  - S3-backed artifact store
  - Model registry with staging/production transitions

- **Backend API**
  - FastAPI service
  - Serves forecast results from database/storage

- **Database**
  - Amazon RDS (PostgreSQL)
  - Stores batch predictions and metadata

- **CI/CD**
  - GitHub Actions
  - Automated testing, container build, and deployment

- **Monitoring**
  - Amazon CloudWatch
  - Feature distribution logging
  - Simple drift detection

---

## 🔁 ML Lifecycle

1. **Data Ingestion**
   - Raw CSVs uploaded to S3

2. **Feature Engineering**
   - Lag features
   - Rolling statistics
   - Holiday indicators

3. **Model Training**
   - XGBoost regression model
   - MLflow experiment logging

4. **Model Registry**
   - Register trained models
   - Promote to `Staging` or `Production`

5. **Batch Inference**
   - Scheduled prediction jobs
   - Forecasts stored in RDS / S3

6. **Serving**
   - FastAPI endpoint to query forecasts

7. **Monitoring & Retraining**
   - Drift detection
   - Trigger retraining pipeline

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|-------|------|------------|
| `/forecast` | GET | Fetch forecast for a store/department |
| `/health` | GET | Service health check |

Example:
    `GET /forecast?store_id=1&dept_id=3&week=2012-11-02`

---

## 📁 Project Structure

```text
.
├── data/
│   ├── raw/
│   ├── processed/
├── training/
│   ├── train.py
│   ├── features.py
│   └── config.yaml
├── inference/
│   ├── batch_predict.py
├── api/
│   ├── main.py
│   └── schemas.py
├── mlflow/
│   └── server/
├── docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.train
├── ci_cd/
│   └── github-actions.yml
├── monitoring/
│   └── drift_checks.py
├── requirements.txt
├── README.md
```

---

## 🚀 Deployment

- Data Layer
- Training & Inference
- Experiment Tracking
- Batch Scheduling
- Backend API
- Database
- Dockerized services
- Deployed on AWS EC2 / ECS
- CI/CD via GitHub Actions

---

## 📈 Monitoring & Drift Detection

- Logs prediction distributions
- Compares weekly feature statistics
- Alerts on significant distribution shifts

---

## 🧪 Future Improvements

- Advanced drift detection (KS-test, PSI)
- Automated retraining triggers
- Airflow / EventBridge orchestration
- Dashboard for forecast visualization
- Feature store integration

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

## Contact

For any questions or inquiries, feel free to contact me at [avindashamal@gmail.com].