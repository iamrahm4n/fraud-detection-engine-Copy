# Fraud Detection & Transaction Risk Engine

A machine learning-powered backend application that analyses financial transactions, estimates fraud probability, assigns risk levels, and provides transaction decisions.

## Project Overview

This project uses machine learning to identify potentially fraudulent transactions using historical credit card transaction data.

The system combines a Random Forest classifier with a configurable risk engine to generate three possible decisions:

- **LOW** — APPROVE
- **MEDIUM** — REVIEW
- **HIGH** — DECLINE

This is an educational prototype, not a production banking system.

## Architecture

Client
  ↓
FastAPI
  ↓
FraudService
  ↓
Random Forest Model
  ↓
RiskEngine
  ↓
Repository
  ↓
SQLite Database

## Machine Learning

Models explored:
- Logistic Regression
- Random Forest Classifier

The final model is a Random Forest classifier.

### Dataset

The project uses the Kaggle Credit Card Fraud Detection dataset.

Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The dataset contains anonymized transaction features and a binary fraud label.

### Final Model Performance

Evaluation on the held-out test dataset:

| Metric | Result |
|---|---:|
| Precision | 94.52% |
| Recall | 72.63% |
| F1 Score | Approximately 82% |
| Average Precision | 0.8012 |

Confusion matrix:

- True negatives: 56,647
- False positives: 4
- False negatives: 26
- True positives: 69

## Risk Engine

The RiskEngine converts fraud probabilities into business decisions.

| Fraud Probability | Risk Level | Decision |
|---|---|---|
| Below 0.20 | LOW | APPROVE |
| 0.20 to below 0.50 | MEDIUM | REVIEW |
| 0.50 or above | HIGH | DECLINE |

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- FastAPI
- Pydantic
- SQLAlchemy
- SQLite
- Joblib
- Pytest

## Project Structure

```text
fraud-detection-engine/
├── data/
├── models/
├── scripts/
├── src/
│   └── fraud_engine/
│       ├── api/
│       ├── database/
│       ├── detection/
│       └── services/
├── tests/
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup Instructions

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
cd fraud-detection-engine
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the Kaggle dataset and place `creditcard.csv` inside:

```text
data/raw/
```

Run the data preparation and training scripts in the appropriate order.

Start the API:

```bash
python -m uvicorn fraud_engine.api.main:app --reload --app-dir src
```

Open the interactive API documentation:

http://127.0.0.1:8000/docs

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/transactions/analyze` | Analyse a transaction |

## Future Improvements

- Improved input validation
- More comprehensive automated testing
- Model monitoring and performance tracking
- Production-grade database integration
- Authentication and authorization

## Disclaimer

This project is developed for learning and demonstration purposes. Model predictions are estimates and should not be treated as guaranteed fraud detection outcomes.
