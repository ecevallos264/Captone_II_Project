# Stock Market Prediction Dashboard

## Project Overview

This project is a machine learning-based stock market prediction system designed to estimate short-term stock price movement using historical market data. The application combines data preprocessing, feature engineering, regression modeling, and visualization to generate predictions for selected stock tickers through an interactive dashboard.

## Team Members

* Gavin Greene
* Erick Cevallos
* Alexander Hernandez
* Ryan Echevarria

## Objectives

The goal of this project is to improve stock price forecasting by integrating multiple financial indicators and evaluating how additional features affect prediction accuracy.

## Features

* Historical stock data retrieval
* Feature engineering using:

  * Closing Price
  * Trading Volume
  * Moving Averages
* Regression-based prediction model
* Confidence interval visualization
* Interactive dashboard for ticker selection
* Batch testing across multiple stocks

## Models Used

### Baseline Model

* Linear Regression

### Additional Model Exploration

* LSTM (Long Short-Term Memory) for sequential forecasting

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Plotly / Dashboard tools
* Docker

## Project Structure

* `data/` → raw and processed stock data
* `models/` → saved model files
* `notebooks/` → experimentation notebooks
* `dashboard/` → visualization interface
* `scripts/` → preprocessing and prediction scripts

## Workflow

1. Retrieve historical stock market data
2. Clean and preprocess dataset
3. Generate financial indicators
4. Train prediction model
5. Evaluate using MAE, RMSE, and R²
6. Visualize predictions against actual values

## Evaluation Metrics

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R² Score

## Current Improvements

* Added Trading Volume as a predictive feature
* Added Moving Average indicators
* Improved batch retrieval pipeline
* Dockerized deployment for reproducibility

## Future Work

* Expand LSTM implementation
* Add more technical indicators
* Improve live API integration
* Increase forecasting horizon

## How to Run

1. Install dependencies
2. Run preprocessing script
3. Train model
4. Launch dashboard

## Notes

This project was developed as part of a capstone course focused on machine learning implementation and team collaboration.
