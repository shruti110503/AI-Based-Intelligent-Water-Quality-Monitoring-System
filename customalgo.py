import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')  # Keeps styling clean but with a white background
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import plotly.offline as pyo
from scipy import stats
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from xgboost.sklearn import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc

# Load Dataset
df = pd.read_csv("water_potability.csv")

# Data Preprocessing
df['Potability'] = df['Potability'].astype('category')
for column in ['ph', 'Sulfate', 'Trihalomethanes']:
    df[column] = df[column].fillna(df.groupby('Potability', observed=False)[column].transform('mean'))

# Train-Test Split
X = df.drop('Potability', axis=1)
y = df['Potability']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Balancing Data using SMOTE
smt = SMOTE()
X_train, y_train = smt.fit_resample(X_train, y_train)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train XGBoost and Random Forest
xgb_model = XGBClassifier()
rf_model = RandomForestClassifier()

xgb_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# Predict Probabilities
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
rf_probs = rf_model.predict_proba(X_test)[:, 1]

# Weighted Ensemble Prediction (70% XGBoost, 30% Random Forest)
ensemble_probs = (0.7 * xgb_probs) + (0.3 * rf_probs)
ensemble_preds = [1 if prob > 0.5 else 0 for prob in ensemble_probs]

# Model Evaluation
print(classification_report(y_test, ensemble_preds))
print(f"Ensemble Accuracy: {accuracy_score(y_test, ensemble_preds)}")

# Confusion Matrix Heatmap
plt.figure(figsize=(6, 4), facecolor='white')  # Ensures a white background
sns.heatmap(confusion_matrix(y_test, ensemble_preds), annot=True, fmt='d', cmap='coolwarm')
plt.title("Confusion Matrix - XGBoost + Random Forest Hybrid", color='black')
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, ensemble_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6), facecolor='white')  # Ensures a white background
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Hybrid Model (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--', lw=2)
plt.xlabel("False Positive Rate", color='black')
plt.ylabel("True Positive Rate", color='black')
plt.title("Receiver Operating Characteristic (ROC) Curve", color='black')
plt.legend(loc="lower right")
plt.show()
