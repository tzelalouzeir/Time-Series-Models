# Time Series Prediction Models

In my journey to explore the vast world of **Machine Learning** and **Deep Learning**, I delved into various time series prediction models.

## Overview
The primary aim was to understand, implement, and evaluate different models on their capability to forecast time series data, specifically stock prices. The `Close` price was predominantly chosen as a feature for model building.

## 📊 What's Inside:

### 1️⃣ Data Preparation
- Split the data into **training** and **test** sets.
- Scaled the data for smoother model ingestion.

### 2️⃣ Prophet Model
- Conducted parameter tuning using grid search to identify the optimal changepoint and seasonality prior scales.
- Utilized the best performing parameters for predictions on the test data.

### 3️⃣ XGBoost
- Employed random search over a broad parameter space for hyperparameter optimization.
- Assessed the model's performance on a hold-out test set.

### 4️⃣ SARIMA & ARIMA
- Implemented these classic time series forecasting methods.
- Benchmarked their performance using metrics like MAE (Mean Absolute Error) and RMSE (Root Mean Squared Error).

### 5️⃣ Linear Regression
- Utilized dates as a single feature to predict the 'Close' prices.
- Analyzed its forecasting power with MAE and RMSE.

## 📌 Note
The code and methods highlighted above offer a simplified representation of the entire process. The insights gleaned from the results were both fascinating and invaluable in understanding stock price behaviors.

## 🤝 Let's Connect!
Connect with me on [LinkedIn](https://www.linkedin.com/in/tzelalouzeir/).

For more insights into my work, check out my latest project: [tafou.io](https://tafou.io).

I'm always eager to learn, share, and collaborate. If you have experiences, insights, or thoughts about RL, Prophet, XGBoost, SARIMA, ARIMA, or even simple Linear Regression in the domain of forecasting, please create an issue, drop a comment, or even better, submit a PR! 

_Let's learn and grow together!_ 🌱
