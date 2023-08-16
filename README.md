# Time Series Prediction Models

In my journey to explore the vast world of **Machine Learning** and **Deep Learning**, I delved into various time series prediction models.

## Overview
The primary aim was to understand, implement, and evaluate different models on their capability to forecast time series data, specifically stock prices. The `Close` price was predominantly chosen as a feature for model building.

## 📊 What's Inside:

### 1️⃣ Data Preparation
- Data contain **'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'** dataframes.
- Split the data into **training** and **test** sets.
- Scaled the data for smoother model ingestion.
<img src="https://github.com/tzelalouzeir/Time-Series-Models/blob/main/1.png" alt="Data Preparation" width="600">

### 2️⃣ Prophet Model
- Conducted parameter tuning using grid search to identify the optimal changepoint and seasonality prior scales.
- Utilized the best performing parameters for predictions on the test data.
<img src="https://github.com/tzelalouzeir/Time-Series-Models/blob/main/2pro.png" alt="Prophet Model" width="600">

### 3️⃣ XGBoost
- Employed random search over a broad parameter space for hyperparameter optimization.
- Assessed the model's performance on a hold-out test set.
<img src="https://github.com/tzelalouzeir/Time-Series-Models/blob/main/3xgb.png" alt="XGBoost" width="600">

### 4️⃣ SARIMA & ARIMA
- Implemented these classic time series forecasting methods.
- Benchmarked their performance using metrics like MAE (Mean Absolute Error) and RMSE (Root Mean Squared Error).

### 5️⃣ Linear Regression
- Utilized dates as a single feature to predict the 'Close' prices.
- Analyzed its forecasting power with MAE and RMSE.
<img src="https://github.com/tzelalouzeir/Time-Series-Models/blob/main/4sar.png" alt="SARIMA & ARIMA and LR" width="600">

## Outputs of Performance:

- **1 Day Data (40 Data):**
<img src="https://github.com/tzelalouzeir/Time-Series-Models/blob/main/5_1day.png" alt="1 Day Data" width="600">

- **12 Hours Data (90 Data):**
<img src="https://github.com/tzelalouzeir/Time-Series-Models/blob/main/6_12hour.png" alt="12 Hours Data" width="600">

- **1 Hour Data (1000 Data):**
<img src="https://github.com/tzelalouzeir/Time-Series-Models/blob/main/7_1hourdata.png" alt="1 Hour Data" width="600">

## 📌 Note
The code and methods highlighted above offer a simplified representation of the entire process. The insights gleaned from the results were both fascinating and invaluable in understanding stock price behaviors.

## 🤝 Let's Connect!
Connect with me on [LinkedIn](https://www.linkedin.com/in/tzelalouzeir/).

For more insights into my work, check out my latest project: [tafou.io](https://tafou.io).

I'm always eager to learn, share, and collaborate. If you have experiences, insights, or thoughts about RL, Prophet, XGBoost, SARIMA, ARIMA, or even simple Linear Regression in the domain of forecasting, please create an issue, drop a comment, or even better, submit a PR! 

_Let's learn and grow together!_ 🌱
