SOLUTION = """
# TITANIC SURVIVAL PREDICTION: FEATURE ENGINEERING & RANDOM FOREST
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

sns.set_theme(style="whitegrid")


# 1. Data Loading

print("Loading Titanic dataset from seaborn...")
df = sns.load_dataset('titanic')
print(f"Dataset loaded: {df.shape[0]} passengers, {df.shape[1]} columns.\\n")


# 2. Exploratory Data Analysis

print("Survival breakdown by class:")
print(df.groupby(['pclass', 'sex'])['survived'].mean().unstack(), "\\n")

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
df['survived'].value_counts().plot(kind='bar', color=['#e74c3c', '#2ecc71'])
plt.title('Overall Survival Counts')
plt.xticks([0, 1], ['Did Not Survive', 'Survived'], rotation=0)

plt.subplot(1, 3, 2)
sns.barplot(x='pclass', y='survived', data=df, palette='Blues_d')
plt.title('Survival Rate by Class')

plt.subplot(1, 3, 3)
sns.barplot(x='sex', y='survived', data=df, palette='Set2')
plt.title('Survival Rate by Gender')

plt.tight_layout()
plt.show()


# 3. Feature Engineering

print("Engineering features...")

df_clean = df.copy()

# Family size as a single feature
df_clean['family_size'] = df_clean['sibsp'] + df_clean['parch'] + 1
df_clean['is_alone'] = (df_clean['family_size'] == 1).astype(int)

# Extract title from name
df_clean['title'] = df_clean['who'].map({'man': 0, 'woman': 1, 'child': 2})

# Fill missing age with median by class and sex
df_clean['age'] = df_clean.groupby(['pclass', 'sex'])['age'].transform(lambda x: x.fillna(x.median()))

# Age bins
df_clean['age_group'] = pd.cut(df_clean['age'], bins=[0, 12, 18, 35, 60, 100],
                               labels=['child', 'teen', 'adult', 'middle_age', 'senior'])

# Fare bins
df_clean['fare_group'] = pd.qcut(df_clean['fare'], q=4, labels=['low', 'mid', 'high', 'premium'])

# Encode categoricals
le = LabelEncoder()
df_clean['sex_enc'] = le.fit_transform(df_clean['sex'])
df_clean['embarked_enc'] = le.fit_transform(df_clean['embarked'].fillna('S'))
df_clean['age_group_enc'] = le.fit_transform(df_clean['age_group'].astype(str))
df_clean['fare_group_enc'] = le.fit_transform(df_clean['fare_group'].astype(str))


# 4. Model Training

features = ['pclass', 'sex_enc', 'age', 'family_size', 'is_alone',
            'fare', 'embarked_enc', 'title', 'age_group_enc', 'fare_group_enc']

X = df_clean[features].fillna(0)
y = df_clean['survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\\nTraining Random Forest classifier...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)


# 5. Evaluation

y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cv_scores = cross_val_score(rf_model, X, y, cv=5)

print("\\n" + "="*50)
print("MODEL PERFORMANCE")
print("="*50)
print(f"Test Accuracy: {accuracy:.4f}")
print(f"5-Fold CV: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
print("\\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Did Not Survive', 'Survived']))

# Confusion matrix
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Not Survived', 'Predicted Survived'],
            yticklabels=['Actual Not Survived', 'Actual Survived'])
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# Feature importance
print("\\nTop Feature Importances:")
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)
print(importance_df.to_string(index=False))
""".strip()

DESCRIPTION = "Predict Titanic survival with Random Forest using feature engineering (family size, title, age/fare bins) and 5-fold cross-validation."
