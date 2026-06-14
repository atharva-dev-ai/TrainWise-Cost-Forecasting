# TrainWise AI

### AI Infrastructure Cost Forecasting Platform

TrainWise AI is an end-to-end Machine Learning project that predicts the cost of training AI models across major cloud providers including AWS, Azure, and Google Cloud Platform (GCP).

The platform helps ML engineers, startups, and AI teams estimate training expenses before launching costly GPU-intensive workloads.

---

## Problem Statement

Training modern AI models can cost anywhere from a few dollars to hundreds of thousands of dollars depending on:

- Cloud Provider
- GPU Type
- GPU Count
- Training Duration
- Model Size
- Dataset Size
- Number of Epochs

Estimating these costs manually is often difficult and error-prone.

TrainWise AI leverages machine learning to forecast training costs and support infrastructure planning.

---

## Project Objectives

- Forecast AI model training costs
- Compare cloud infrastructure expenses
- Improve budgeting for AI workloads
- Demonstrate an end-to-end ML deployment workflow

---

## Dataset

A synthetic dataset containing **100,000 training jobs** was generated using realistic cloud pricing assumptions.

### Features

| Feature | Description |
|----------|-------------|
| cloud_provider | AWS, Azure, or GCP |
| gpu_type | T4, A10G, A100, H100 |
| gpu_count | Number of GPUs used |
| training_hours | Total training duration |
| model_parameters_billion | Model size in billions of parameters |
| dataset_size_gb | Dataset size in GB |
| epochs | Number of training epochs |

### Target Variable

| Target |
|----------|
| training_cost_usd |

---

## Machine Learning Pipeline

### Data Processing

- Train-Test Split
- One-Hot Encoding
- Feature Alignment

### Model Development

- Linear Regression (Baseline)
- XGBoost Regression (Final Model)

### Evaluation Metrics

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score
- 5-Fold Cross Validation

---

## Model Performance

| Metric | Score |
|----------|----------|
| Test R² | 0.927 |
| Cross Validation R² | 0.914 |
| MAE | 219 USD |
| RMSE | 940 USD |

The XGBoost model demonstrated strong predictive performance and was selected as the final production model.

---

## Technology Stack

### Machine Learning

- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost

### Visualization

- Matplotlib

### Version Control

- Git
- GitHub

### Deployment (Upcoming)

- Streamlit
- Docker
- AWS EC2

---

## Repository Structure

```text
TrainWise-Cost-Forecasting/

├── TrainWise_CloudCost_Forecasting.ipynb
├── trainwise_ai_dataset.csv
├── trainwise_model.pkl
├── feature_columns.pkl
├── LICENSE
└── README.md
```

---

## Future Enhancements

- Interactive Streamlit Dashboard
- Real-Time Cost Estimation
- Cost Optimization Recommendations
- Docker Containerization
- AWS Cloud Deployment
- Multi-Cloud Cost Comparison

---

## Author

Atharva

Machine Learning • Data Science • AI Products

---

## Project Status

🚀 Active Development

Current Phase:
Machine Learning Model Development Completed

Next Phase:
Streamlit Application Development and AWS Deployment
