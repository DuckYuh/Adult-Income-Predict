import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# 1. Load dataset
data = pd.read_csv("adult.csv")

data = data.sample(n=5000, random_state=42)

# 2. Prepare data
# Encode categorical variables
le = LabelEncoder()
categorical_columns = data.select_dtypes(include=['object']).columns

for col in categorical_columns:
    if col != 'income':  # Don't encode target yet
        data[col] = le.fit_transform(data[col].astype(str))

# Separate features and target
X = data.drop("income", axis=1)
y = le.fit_transform(data["income"])

# 3. Split Train/Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Train Decision Tree model
clf = DecisionTreeClassifier(criterion="entropy", max_depth=6, random_state=42)
clf.fit(X_train, y_train)

# 5. Predictions
y_pred = clf.predict(X_test)

# 6. Evaluate model
print("Accuracy (Train):", accuracy_score(y_train, clf.predict(X_train)))
print("Accuracy (Test):", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['<=50K', '>50K']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['<=50K', '>50K'], yticklabels=['<=50K', '>50K'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# 7. Feature Importance
importance = clf.feature_importances_
feature_names = X.columns
feat_imp = pd.DataFrame({"Feature": feature_names, "Importance": importance})
feat_imp = feat_imp.sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(data=feat_imp.head(15), x="Importance", y="Feature")
plt.title("Top Feature Importance")
plt.tight_layout()
plt.show()

# 8. Plot Decision Tree (simplified view due to many features)
plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=X.columns, class_names=['<=50K', '>50K'],
          filled=True, rounded=True, fontsize=8)
plt.title("Decision Tree Structure")
plt.tight_layout()
plt.show()