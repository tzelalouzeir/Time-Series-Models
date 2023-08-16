# Split the data into training and testing sets
train_size = int(len(df) * 0.8)
train_data = df.iloc[:train_size, :]
test_data = df.iloc[train_size:, :]
print("Train Data:", len(train_data))
print("Test Data:", len(test_data))

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_data = scaler.fit_transform(train_data)
scaled_test_data = scaler.transform(test_data)

#################################### Prophet ####################################

# Define the parameter grid for Prophet
param_grid = {
    'changepoint_prior_scale': [0.005, 0.01, 0.1, 1.0],
    'seasonality_prior_scale': [0.005, 0.01, 0.1, 1.0]
}

# Create a Prophet model
prophet_model = Prophet()

# Prepare the data for Prophet
prophet_data = pd.DataFrame(train_data['Close'].values, index=train_data.index, columns=['y'])
prophet_data['ds'] = prophet_data.index

best_score = np.inf
best_params = None

# Iterate over parameter combinations
for changepoint_prior in param_grid['changepoint_prior_scale']:
    for seasonality_prior in param_grid['seasonality_prior_scale']:
        # Set the parameters for the Prophet model
        prophet_model = Prophet(changepoint_prior_scale=changepoint_prior, seasonality_prior_scale=seasonality_prior)

        # Fit the model
        prophet_model.fit(prophet_data)

        # Make predictions
        prophet_future = prophet_model.make_future_dataframe(periods=len(test_data))
        prophet_predictions = prophet_model.predict(prophet_future)['yhat'].tail(len(test_data)).values

        # Evaluate the model using a scoring metric (e.g., mean squared error)
        score = mean_squared_error(test_data['Close'].values, prophet_predictions)

        # Check if the current combination of parameters gives a better score
        if score < best_score:
            best_score = score
            best_params = {'changepoint_prior_scale': changepoint_prior, 'seasonality_prior_scale': seasonality_prior}

# Update the Prophet model with the best parameters
prophet_model = Prophet(changepoint_prior_scale=best_params['changepoint_prior_scale'], seasonality_prior_scale=best_params['seasonality_prior_scale'])

# Fit the model with the best parameters
prophet_model.fit(prophet_data)

# Make final predictions with the best model
prophet_future = prophet_model.make_future_dataframe(periods=len(test_data))
prophet_predictions = prophet_model.predict(prophet_future)['yhat'].tail(len(test_data)).values
final_mse = mean_squared_error(test_data['Close'].values, prophet_predictions)
print(f"\nMSE of the best Prophet model on test data: {final_mse:.2f}")

#################################### XGBoost ####################################

# Define the parameter grid for XGBoost
param_grid = {
    'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3],
    'max_depth': [3, 5, 7, 9, 11],
    'n_estimators': [50, 100, 150, 200, 300, 400, 500],
    'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0],
    'alpha': [0, 0.1, 0.5, 1.0],
    'lambda': [0, 0.1, 0.5, 1.0]
}

# Create an XGBoost regressor
xgb_model = XGBRegressor(objective='reg:squarederror')

# Perform random search
random_search = RandomizedSearchCV(xgb_model, param_distributions=param_grid, n_iter=100, cv=5, verbose=0, random_state=42, n_jobs=-1)
random_search.fit(scaled_train_data, train_data['Close'].values)

# Make predictions with the best model
xgb_predictions = random_search.best_estimator_.predict(scaled_test_data[-7:])
true_values = test_data['Close'].values[-7:]

# Calculate and print performance metrics
mae = mean_absolute_error(true_values, xgb_predictions)
mse = mean_squared_error(true_values, xgb_predictions)
r2 = r2_score(true_values, xgb_predictions)

print(f"\nBest Model Parameters XGBoost: {random_search.best_params_}")
print(f"XGBoost Mean Absolute Error (MAE): {mae:.2f}")
print(f"XGBoost Mean Squared Error (MSE): {mse:.2f}")
print(f"XGBoost R^2 Score: {r2:.2f}\n")

#################################### SARIMA & ARIMA & LR ####################################

# Define the SARIMA model
sarima_model = SARIMAX(train_data['Close'].values, order=(1, 0, 0), seasonal_order=(1, 1, 1, 12), enforce_invertibility=False)
sarima_model_fit = sarima_model.fit(disp=False)

# Make predictions with the SARIMA model
sarima_predictions = sarima_model_fit.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1)

# Define the ARIMA model
arima_model = ARIMA(train_data['Close'].values, order=(1, 0, 0))
arima_model_fit = arima_model.fit()

# Make predictions with the ARIMA model
arima_predictions = arima_model_fit.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1)

# Define a function to compute RMSE
def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)

# Compute performance metrics for SARIMA
sarima_mae = mean_absolute_error(test_data['Close'].values, sarima_predictions)
sarima_rmse = rmse(test_data['Close'].values, sarima_predictions)

# Compute performance metrics for ARIMA
arima_mae = mean_absolute_error(test_data['Close'].values, arima_predictions)
arima_rmse = rmse(test_data['Close'].values, arima_predictions)

# Transform date to a numeric feature
train_dates = np.array(range(len(train_data))).reshape(-1, 1)
test_dates = np.array(range(len(train_data), len(df))).reshape(-1, 1)

# Scale the data
scaler_dates = MinMaxScaler(feature_range=(0, 1))
scaled_train_dates = scaler_dates.fit_transform(train_dates)
scaled_test_dates = scaler_dates.transform(test_dates)

# Train the Linear Regression model using date as feature
lr_model = LinearRegression()
lr_model.fit(scaled_train_dates, train_data['Close'].values)

# Make predictions with the Linear Regression model
lr_predictions = lr_model.predict(scaled_test_dates)

# Compute performance metrics for Linear Regression
lr_mae = mean_absolute_error(test_data['Close'].values, lr_predictions)
lr_rmse = rmse(test_data['Close'].values, lr_predictions)

# Print the performance metrics
print("\nSARIMA Performance:")
print(f"MAE: {sarima_mae:.2f}")
print(f"RMSE: {sarima_rmse:.2f}")

print("\nARIMA Performance:")
print(f"MAE: {arima_mae:.2f}")
print(f"RMSE: {arima_rmse:.2f}")

print("\nLinear Regression Performance:")
print(f"MAE: {lr_mae:.2f}")
print(f"RMSE: {lr_rmse:.2f}")
