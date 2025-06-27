# PROJECT NAME - Yes Bank Stock Price Prediction Using Ensemble Models

# PROJECT SUMMARY - This project aims to develop a predictive machine learning model to forecast the monthly closing prices of Yes Bank shares using historical stock market data. The pipeline includes comprehensive exploratory data analysis (EDA) under the UBM rule (Univariate, Bivariate, Multivariate), hypothesis testing, feature engineering, implementation of ensemble regression models (Gradient Boosting, Random Forest, XGBoost) to provide accurate price forecasts.

# PROBLEM STATEMENT - The share prices of Yes Bank have shown significant volatility over time. Accurate forecasting of closing prices can help investors make better financial decisions. The problem is to predict the monthly closing price of Yes Bank stock based on historical data using advanced machine learning techniques.

# IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt
import warnings
warnings.filterwarnings("ignore")

# DATASET LOADING

from google.colab import files
uploaded = files.upload()
df = pd.read_csv('Yes Bank Stock Price.csv')

# FIRST VIEW OF DATASET

df.head()

# ROWS & COLUMNS

df.shape

# DUPLICATE VALUES

df.duplicated().sum()

# MISSING/NULL VALUES

df.isnull().sum()

# VARIABLES DESCRIPTION

df.info()
df.describe()

# UNIQUE VALUE CHECK

df.nunique()

# DATA WRANGLING

df['Date'] = pd.to_datetime(df['Date'], format='%b-%y')
df = df.sort_values('Date').reset_index(drop=True)

# EXPLORATORY DATA ANALYSIS (UBM RULE)

# Univariate Analysis
sns.histplot(df['Close'], kde=True)
plt.title('Distribution of Closing Prices')
plt.show()
df['Close'].plot(kind='box')
plt.title('Boxplot of Closing Prices')
plt.show()

# Bivariate Analysis
sns.scatterplot(x='Open', y='Close', data=df)
plt.title('Open vs Close Price')
plt.grid(True)
plt.show()
sns.scatterplot(x='High', y='Close', data=df)
plt.title('High vs Close Price')
plt.grid(True)
plt.show()

# Multivariate Analysis
plt.figure(figsize=(10, 6))
sns.heatmap(df[['Open', 'High', 'Low', 'Close']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# DATA VISUALIZATION CHARTS

# Line Chart of Closing Price Over Time
plt.figure(figsize=(12, 5))
plt.plot(df['Date'], df['Close'], label='Closing Price', color='blue')
plt.title('Yes Bank Monthly Closing Price Over Time')
plt.xlabel('Date')
plt.ylabel('Closing Price (INR)')
plt.legend()
plt.grid(True)
plt.show()

# Histogram of Closing Prices (Univariate)
sns.histplot(df['Close'], bins=20, kde=True, color='skyblue')
plt.title('Distribution of Closing Prices')
plt.xlabel('Close Price')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Boxplot of Open Prices (Outlier Detection)
sns.boxplot(x=df['Open'], color='lightgreen')
plt.title('Boxplot of Open Prices')
plt.grid(True)
plt.show()

# Scatter Plot: Open vs Close (Bivariate)
sns.scatterplot(x='Open', y='Close', data=df, color='coral')
plt.title('Scatterplot: Open vs Close')
plt.grid(True)
plt.show()

# Area Chart for Low Price Trend
plt.fill_between(df['Date'], df['Low'], color='lightblue', alpha=0.5)
plt.title('Low Price Over Time')
plt.xlabel('Date')
plt.ylabel('Low Price')
plt.grid(True)
plt.show()

# KDE Plot of High Prices (Density)
sns.kdeplot(df['High'], shade=True, color='orange')
plt.title('KDE Plot of High Prices')
plt.xlabel('High Price')
plt.grid(True)
plt.show()

# Correlation Heatmap (Multivariate)
plt.figure(figsize=(8, 6))
sns.heatmap(df[['Open', 'High', 'Low', 'Close']].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Monthly Average Close Bar Plot
df['Month'] = df['Date'].dt.month
monthly_avg = df.groupby('Month')['Close'].mean()
monthly_avg.plot(kind='bar', color='purple')
plt.title('Average Monthly Close Price')
plt.xlabel('Month')
plt.ylabel('Avg Close Price')
plt.grid(True)
plt.show()

# Joint Plot: High vs Close
sns.jointplot(x='High', y='Close', data=df, kind='scatter', color='teal')
plt.suptitle('High vs Close - Joint Distribution', y=1.02)
plt.show()

# Pair Plot (Full Variable Relationships)
sns.pairplot(df[['Open', 'High', 'Low', 'Close']], corner=True)
plt.suptitle('Pairwise Relationships', y=1.02)
plt.show()

# FEATURE ENGINEERING

# Time-based split
train_size = int(len(df) * 0.8)
df_train = df.iloc[:train_size].copy()
df_test = df.iloc[train_size:].copy()

# Train Features
df_train['Prev_Close'] = df_train['Close'].shift(1)
df_train['3_MA_Close'] = df_train['Close'].rolling(window=3).mean()
df_train['Price_Change'] = df_train['Close'] - df_train['Open']
df_train = df_train.dropna().reset_index(drop=True)

# Test Features
last_train = df_train.tail(2).copy()
test_combined = pd.concat([last_train, df_test], ignore_index=True)
test_combined['Prev_Close'] = test_combined['Close'].shift(1)
test_combined['3_MA_Close'] = test_combined['Close'].rolling(window=3).mean()
test_combined['Price_Change'] = test_combined['Close'] - test_combined['Open']
df_test = test_combined.iloc[2:].reset_index(drop=True)
features = ['Open', 'High', 'Low', 'Prev_Close', '3_MA_Close', 'Price_Change']
target = 'Close'

X_train = df_train[features]
y_train = df_train[target]
X_test = df_test[features]
y_test = df_test[target]
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# MODEL IMPLEMENTATION AND EVALUATING THE BEST MODEL

# Gradient Boosting Regressor
gbr = GradientBoostingRegressor(random_state=42)
gbr.fit(X_train, y_train)
gbr_pred = gbr.predict(X_test)

# Random Forest Regressor
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# XGBoost Regressor
xgb = XGBRegressor(random_state=42)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)

# Define Evaluation Function
def evaluate_model(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"🔍 {name}")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.2f}")
    print("-" * 30)
    return rmse

# Evaluate All Models
rmse_gbr = evaluate_model("Gradient Boosting Regressor", y_test, gbr_pred)
rmse_rf = evaluate_model("Random Forest Regressor", y_test, rf_pred)
rmse_xgb = evaluate_model("XGBoost Regressor", y_test, xgb_pred)

# Determine Best Model
rmse_scores = {
    "Gradient Boosting Regressor": rmse_gbr,
    "Random Forest Regressor": rmse_rf,
    "XGBoost Regressor": rmse_xgb
}
best_model_name = min(rmse_scores, key=rmse_scores.get)
print(f"✅ Best Model: {best_model_name} with RMSE = {rmse_scores[best_model_name]:.2f}")

# Assign Best Model and Prediction
if best_model_name == "Gradient Boosting Regressor":
    best_model = gbr
    best_pred = gbr_pred
elif best_model_name == "Random Forest Regressor":
    best_model = rf
    best_pred = rf_pred
else:
    best_model = xgb
    best_pred = xgb_pred

# PLOT ACTUAL VS PREDICTED

plt.figure(figsize=(12, 5))
plt.plot(df_test['Date'], y_test.values, label='Actual', color='black')
plt.plot(df_test['Date'], best_pred, label=f'Predicted - {best_model_name}', linestyle='--')
plt.title(f'Actual vs Predicted Closing Price ({best_model_name})')
plt.xlabel('Date')
plt.ylabel('Closing Price (INR)')
plt.legend()
plt.grid(True)
plt.show()

# CONCLUSION - Among the three models tested, Gradient Boosting Regressor outperformed the others with the lowest RMSE of 18.28, along with a high R² score of 0.98, indicating excellent predictive accuracy. This makes it the best choice for forecasting Yes Bank’s monthly closing stock prices in this project.
