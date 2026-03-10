import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Create static/images if it doesn't exist
os.makedirs('static', exist_ok=True)
os.makedirs('static/images', exist_ok=True)

print("Loading data...")
data = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Configure aesthetics for the charts to match the frontend UI
plt.style.use('dark_background')
plt.rcParams.update({
    "figure.facecolor": "#0f172a",
    "axes.facecolor": "#1e293b", # Removed alpha for matplotlib compatibility
    "axes.edgecolor": "#334155",
    "axes.labelcolor": "#f8fafc",
    "text.color": "#f8fafc",
    "xtick.color": "#94a3b8",
    "ytick.color": "#94a3b8"
})

# 1. Churn Distribution Plot
print("Generating Churn Distribution...")
plt.figure(figsize=(6, 4))
sns.countplot(x='Churn', data=data, palette=['#6366f1', '#ec4899'])
plt.title("Customer Churn Distribution", pad=15)
plt.savefig('static/images/churn_distribution.png', bbox_inches='tight', transparent=True, dpi=300)
plt.close()

# 2. Monthly Charges vs Churn Plot
print("Generating Monthly Charges vs Churn...")
plt.figure(figsize=(8, 5))
sns.boxplot(x='Churn', y='MonthlyCharges', data=data, palette=['#6366f1', '#ec4899'])
plt.title("Monthly Charges vs Churn", pad=15)
plt.savefig('static/images/monthly_charges_churn.png', bbox_inches='tight', transparent=True, dpi=300)
plt.close()

# Load Model for Feature Importance
print("Loading model for Feature Importance...")
bagging_model = joblib.load('model.pkl')
features_list = joblib.load('features.pkl')

print("Generating Feature Importance...")
# Calculate average feature importance across all base estimators
importances = np.mean([tree.feature_importances_ for tree in bagging_model.estimators_], axis=0)

# Sort features by importance
indices = np.argsort(importances)
sorted_features = [features_list[i] for i in indices]
sorted_importances = importances[indices]

# Take top 10 features for cleaner visualization
top_k = 10
sorted_features = sorted_features[-top_k:]
sorted_importances = sorted_importances[-top_k:]

plt.figure(figsize=(10, 6))
plt.barh(sorted_features, sorted_importances, color='#8b5cf6')
plt.title("Top 10 Feature Importances", pad=15)
plt.xlabel("Average Gini Importance")
plt.tight_layout()
plt.savefig('static/images/feature_importance.png', bbox_inches='tight', transparent=True, dpi=300)
plt.close()

print("Generating Correlation Heatmap (Numeric Only)...")
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors='coerce').fillna(0)
numeric_data = data.select_dtypes(include=[np.number])

plt.figure(figsize=(10, 8))
sns.heatmap(numeric_data.corr(), cmap="coolwarm", annot=False, fmt=".2f", linewidths=0.5, linecolor=(1, 1, 1, 0.1))
plt.title("Numeric Feature Correlation Heatmap", pad=15)
plt.tight_layout()
plt.savefig('static/images/correlation_heatmap.png', bbox_inches='tight', transparent=True, dpi=300)
plt.close()

print("All visual graphs generated successfully!")
