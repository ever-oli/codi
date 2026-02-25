SOLUTION = """
# CHURN PREDICTION: CLASS IMBALANCE (SMOTE) & LIGHTGBM

# !pip install datasets imbalanced-learn lightgbm scikit-learn pandas matplotlib seaborn -q

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, precision_recall_curve, auc, confusion_matrix
from imblearn.over_sampling import SMOTE
import lightgbm as lgb

sns.set_theme(style="whitegrid")


# 1. Data Ingestion & Cleaning

print("Loading and cleaning dataset...")
dataset = load_dataset("scikit-learn/churn-prediction", split="train")
df = dataset.to_pandas()

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', np.nan))
df.dropna(inplace=True)
df.drop('customerID', axis=1, inplace=True)

print(f"Data ready. Shape: {df.shape}")


# 2. Feature Engineering & Encoding

print("Encoding categorical variables...")
X = df.drop('Churn', axis=1)
y = df['Churn'].map({'No': 0, 'Yes': 1})

X = pd.get_dummies(X, drop_first=True)


# 3. Train/Test Split & SMOTE

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\\nBefore SMOTE - Training Churn Counts: \\n{y_train.value_counts()}")

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"After SMOTE - Training Churn Counts: \\n{y_train_smote.value_counts()}\\n")

scaler = StandardScaler()
X_train_smote = scaler.fit_transform(X_train_smote)
X_test = scaler.transform(X_test)


# 4. Model Training (LightGBM)

print("Training LightGBM Classifier...")
lgb_model = lgb.LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    random_state=42,
    n_jobs=-1
)

lgb_model.fit(X_train_smote, y_train_smote)


# 5. Model Evaluation

print("\\nGenerating predictions and evaluating...")
y_pred = lgb_model.predict(X_test)
y_prob = lgb_model.predict_proba(X_test)[:, 1]

print("=" * 50)
print("CLASSIFICATION REPORT")
print("=" * 50)
print(classification_report(y_test, y_pred, target_names=['Stayed (0)', 'Churned (1)']))


# 6. Precision-Recall Curve Visualization

precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
pr_auc = auc(recall, precision)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(recall, precision, color='purple', lw=2, label=f'PR Curve (AUC = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")

plt.subplot(1, 2, 2)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', cbar=False,
            xticklabels=['Predicted Stay', 'Predicted Churn'],
            yticklabels=['Actual Stay', 'Actual Churn'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

plt.tight_layout()
plt.show()

print("\\nAnalysis Complete.")
""".strip()

DESCRIPTION = "Predict customer churn using SMOTE for class imbalance handling and LightGBM, evaluated with a precision-recall curve."
