import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
plt.style.use('dark_background')
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import plotly.offline as pyo
from scipy import stats
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from xgboost.sklearn import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc

# Load Dataset
path = "water_potability.csv"
df = pd.read_csv(path)

# Data Preprocessing
df['Potability'] = df['Potability'].astype('category')

# Handling Missing Values (Using observed=False)
for column in ['ph', 'Sulfate', 'Trihalomethanes']:
    df[column] = df[column].fillna(df.groupby('Potability', observed=False)[column].transform('mean'))

# Train-Test Split
#  0-Not Potable, 1-Potable
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

# Model Training and Evaluation
models = [LogisticRegression(), DecisionTreeClassifier(), GaussianNB(), RandomForestClassifier(),
          svm.LinearSVC(), XGBClassifier()]
train_accuracy, test_accuracy = [], []
kfold = KFold(n_splits=10, random_state=7, shuffle=True)

for model in models:
    train_result = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=kfold)
    train_accuracy.append(train_result.mean())
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    test_result = accuracy_score(y_test, y_pred)
    test_accuracy.append(test_result)

# Model Accuracy Comparison 
model_names = ["LogisticRegression", "DecisionTree", "GaussianNB", "RandomForest", "LinearSVC", "XGBoost"]
plt.figure(figsize=(10, 5))
sns.barplot(x=model_names, y=test_accuracy, palette='viridis')
plt.xlabel("Model Name")
plt.ylabel("Test Accuracy")
plt.title("Model Accuracy Comparison")
plt.xticks(rotation=45)
plt.show()

# Best Model: Random Forest
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
y_rfc = rfc.predict(X_test)
print(classification_report(y_test, y_rfc))
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_rfc)}")

# Confusion Matrix Heatmap
sns.heatmap(confusion_matrix(y_test, y_rfc), annot=True, fmt='d', cmap='coolwarm')
plt.title("Confusion Matrix - Random Forest")
plt.show()

# Water Potability Distribution Pie Chart
potability_counts = df['Potability'].value_counts().reset_index()
potability_counts.columns = ['Potability', 'Count']
fig = px.pie(potability_counts, names='Potability', values='Count', 
             title="Water Potability Distribution", hole=0.4, 
             template="plotly_dark")
fig.show()

# Violin Plot 
sns.violinplot(x='Potability', y='ph', data=df, hue='Potability', palette='rocket', legend=False)
plt.title("pH Distribution by Potability")
plt.show()

# Compute ROC Curve for Random Forest (Best Model)
y_probs = rfc.predict_proba(X_test)[:, 1]  # Get probability scores for class 1
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

# Plotting ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Random Forest (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--', lw=2)  # Diagonal line for reference
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.show()


# Create DataFrame with Model Names and Accuracy Scores
model_data = {
    "Model": ["LogisticRegression", "DecisionTree", "GaussianNB", "RandomForest", "LinearSVC", "XGBoost"],
    "Train Accuracy": train_accuracy,
    "Test Accuracy": test_accuracy
}

df_results = pd.DataFrame(model_data)

# Save to Excel File
excel_path = "model_accuracy.xlsx"
df_results.to_excel(excel_path, index=False)

print(f"Excel file saved successfully: {excel_path}")


# ### Conclusion
#    - The Solid levels seem to contain some descripency since its values are on an average 40 folds more than the upper limit for safe drinking water.(Desirable limit for TDS is 500 mg/l and maximum limit is 1000 mg/l which prescribed for drinking purpose.)
# 
#    - The data contains almost equal number of acidic and basic pH level water samples.
# 
#    - The correlation coefficients between the features were very low.
# 
#    - Random Forest and XGBoost worked the best to train the model, both gives us f1 score (Balanced with precision & recall) as around 76%.



