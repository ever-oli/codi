SOLUTION = """
# IRIS DATASET CLASSIFICATION & ALGORITHM COMPARISON
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

sns.set_theme(style="ticks")


# Data Loading

print("Loading the Iris dataset...")
iris = load_iris()

df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

print(f"Dataset loaded successfully with {df.shape[0]} samples.")
print(f"Features: {', '.join(iris.feature_names)}\\n")

# Feature Visualization

print("Generating feature scatter plots...")
g = sns.pairplot(df, hue="species", palette="colorblind", markers=["o", "s", "D"])
g.fig.suptitle("Scatter Plots of Iris Features by Species", y=1.02)
plt.show()


X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Data split into training and testing sets.\\n")


# Model Training and Evaluation

print("Training models and evaluating accuracy...\\n")

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
    "Support Vector Machine (SVM)": SVC(random_state=42, kernel='linear')
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    results[name] = accuracy
    print(f"{name} trained.")


# Final Comparison

print("\\n" + "="*40)
print("FINAL ACCURACY COMPARISON")
print("="*40)

sorted_results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))

for name, acc in sorted_results.items():
    print(f"{name:<30}: {acc * 100:.2f}%")
print("="*40)
""".strip()

DESCRIPTION = "Compare Decision Tree, Random Forest, and SVM classifiers on the Iris dataset with pairplot visualization."
