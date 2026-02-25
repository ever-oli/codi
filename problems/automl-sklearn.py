SOLUTION = """
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# 1. Data Preparation
print("Loading dataset...")
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify feature types
numeric_features = X.columns.tolist()
# Note: California housing is all numeric, but we define the logic for categorical too
categorical_features = []

# 2. Building Preprocessing Transformers
# We automate the handling of missing values and feature scaling
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine transformers into a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 3. Define the AutoML Pipeline
# We start with a placeholder regressor that GridSearchCV will swap out
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor())
])

# 4. Automating Model Selection and Hyperparameter Tuning
# The double underscore notation (regressor__parameter) targets the specific step
param_grid = [
    {
        'regressor': [RandomForestRegressor(random_state=42)],
        'regressor__n_estimators': [50, 100],
        'regressor__max_depth': [None, 10]
    },
    {
        'regressor': [GradientBoostingRegressor(random_state=42)],
        'regressor__n_estimators': [100, 200],
        'regressor__learning_rate': [0.05, 0.1]
    }
]

# 5. Execute the Grid Search
print("Starting AutoML Search...")
grid_search = GridSearchCV(full_pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# 6. Results and Evaluation
print("\\n" + "="*30)
print(f"Best Model Found: {grid_search.best_params_['regressor']}")
print(f"Best CV R2 Score: {grid_search.best_score_:.4f}")
print("="*30)

# Final test set evaluation
final_score = grid_search.score(X_test, y_test)
print(f"Final Test Set R2 Accuracy: {final_score:.4f}")
""".strip()

DESCRIPTION = "Build an AutoML pipeline with sklearn that automates preprocessing, model selection, and hyperparameter tuning via GridSearchCV."
