SOLUTION = """
# HOUSING PRICE REGRESSION & FEATURE SELECTION WITH XGBOOST

# Run this cell to install xgboost if it is not already in the environment:
# !pip install xgboost pandas matplotlib seaborn scikit-learn -q

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

sns.set_theme(style="whitegrid")


# 1. Data Loading and Initial Exploration

print("Loading California Housing dataset...")
california = fetch_california_housing()
df = pd.DataFrame(california.data, columns=california.feature_names)
df['MedHouseVal'] = california.target

print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns.\\n")


# 2. Pre-Modeling Feature Selection (Correlation Matrix)

print("Analyzing linear correlations...")
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title('Feature Correlation Matrix')
plt.show()


# 3. Data Preprocessing

X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 4. Model Training (XGBoost Regressor)

print("\\nTraining XGBoost Regressor...")
xg_reg = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=150,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

xg_reg.fit(X_train, y_train)


# 5. Evaluation Metrics

y_pred = xg_reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("-" * 40)
print("MODEL PERFORMANCE:")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R2):                 {r2:.4f}")
print("-" * 40 + "\\n")


# 6. Feature Importance & Selection

print("Generating Feature Importance plot...")

importance_type = 'gain'
importances = xg_reg.get_booster().get_score(importance_type=importance_type)

importance_df = pd.DataFrame({
    'Feature': list(importances.keys()),
    'Importance (Gain)': list(importances.values())
}).sort_values(by='Importance (Gain)', ascending=True)

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance (Gain)'], color='#3498db')
plt.xlabel('F-Score (Gain)')
plt.ylabel('Features')
plt.title('XGBoost Feature Importance (By Information Gain)')
plt.show()

print("\\nAnalysis Complete.")
""".strip()

DESCRIPTION = "Train an XGBoost regressor on the California Housing dataset and visualize feature importance by information gain."
