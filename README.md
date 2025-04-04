# AI-Based-Intelligent-Water-Quality-Monitoring-System

## 📌 Overview
This project analyzes **water potability** using **machine learning algorithms**. It implements various classification models like **XGBoost, Random Forest, and a custom hybrid model** that integrates Large Language Models (LLMs) for enhanced predictions. The dataset includes parameters like pH, Hardness, Solids, Chloramines, and more.

## 🚀 Features
- **Data Preprocessing**: Handles missing values and balances data using **SMOTE**.
- **Machine Learning Models**:
  - Logistic Regression
  - Decision Tree
  - Gaussian Naïve Bayes
  - Random Forest
  - Support Vector Machine (SVM)
  - XGBoost
  - **Custom Hybrid Model** (XGBoost + Random Forest + LLM)
- **Evaluation Metrics**:
  - Accuracy, Precision, Recall, F1-Score
  - Confusion Matrix
  - ROC Curve
- **Visualizations**:
  - Model Accuracy Comparison (Bar Chart)
  - Water Potability Distribution (Pie Chart)
  - Feature Distributions (Violin Plot)
- **Excel Export**: Saves results as an **Excel file** for further analysis.

## 📂 Dataset
- **Source**: `water_potability.csv`
- **Columns**: pH, Hardness, Solids, Sulfate, Conductivity, Organic Carbon, Trihalomethanes, Turbidity, Potability

## 🛠 Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/water-quality-analysis.git
   cd water-quality-analysis
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure **`openpyxl`** is installed for Excel export:
   ```bash
   pip install openpyxl
   ```

## 🔧 Usage
Run the script to train models and evaluate results:
```bash
python customalgo.py
```

## 📊 Results
- The **hybrid model (XGBoost + Random Forest + LLM)** achieved the highest accuracy.
- The ROC curve and confusion matrix visualize performance.
- Results are saved in **Excel format** (`model_results.xlsx`).

## 🖼 Sample Visualizations
- **Confusion Matrix**
- **ROC Curve**
- **Feature Distributions**

## 🏆 Future Enhancements
- Implementing **real-time water quality monitoring**
- Integrating with **IoT sensors** for live data collection
- Deploying the model as a **web application**

## 📜 License
This project is licensed under the **MIT License**.

---
🔗 **Contributions Welcome!** Feel free to fork, star ⭐, and raise issues! 😊

