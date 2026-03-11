import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# Create static/images if it doesn't exist
os.makedirs('static', exist_ok=True)
os.makedirs('static/images', exist_ok=True)

print("Loading dataset & plot data...")
data = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

with open('custom_plot_data.json', 'r') as f:
    plot_data = json.load(f)

# Configure aesthetics for the charts to match the frontend UI
plt.style.use('dark_background')
plt.rcParams.update({
    "figure.facecolor": "#0f172a",
    "axes.facecolor": "#1e293b",
    "axes.edgecolor": "#334155",
    "axes.labelcolor": "#f8fafc",
    "text.color": "#f8fafc",
    "xtick.color": "#94a3b8",
    "ytick.color": "#94a3b8"
})

# 1. Churn Distribution Plot
print("Generating Churn Distribution...")
plt.figure(figsize=(6, 4))
sns.countplot(x='Churn', hue='Churn', data=data, palette=['#94a3b8', '#0ea5e9'], legend=False)
plt.title("Customer Churn Distribution", pad=15)
plt.savefig('static/images/churn_distribution.png', bbox_inches='tight', transparent=True, dpi=300)
plt.close()

# 2. Confusion Matrix Heatmap
print("Generating Confusion Matrix Heatmap...")
cm = plot_data["confusion_matrix"]
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Churn', 'Churn'], 
            yticklabels=['No Churn', 'Churn'],
            linecolor=(1, 1, 1, 0.1), linewidths=0.5)
plt.title("Confusion Matrix Heatmap", pad=15)
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.savefig('static/images/confusion_matrix.png', bbox_inches='tight', transparent=True, dpi=300)
plt.close()

# 3. Accuracy vs Number of Trees Graph
print("Generating Accuracy vs Trees Graph...")
acc_vs_trees = plot_data["acc_vs_trees"]
num_trees = list(range(1, len(acc_vs_trees) + 1))
plt.figure(figsize=(8, 5))
plt.plot(num_trees, acc_vs_trees, color='#e2e8f0', marker='o', linewidth=2)
plt.title("Ensemble Accuracy vs Number of Trees", pad=15)
plt.xlabel("Number of Decision Trees")
plt.ylabel("Testing Accuracy")
plt.xticks(num_trees)
plt.grid(color='#334155', linestyle='--', linewidth=0.5)
plt.savefig('static/images/accuracy_vs_trees.png', bbox_inches='tight', transparent=True, dpi=300)
plt.close()

# 4. Precision Recall Curve
print("Generating Precision-Recall Curve...")
pr_curve = plot_data["pr_curve"]
precisions = [p["precision"] for p in pr_curve]
recalls = [p["recall"] for p in pr_curve]
plt.figure(figsize=(8, 5))
plt.plot(recalls, precisions, color='#38bdf8', linewidth=2, marker='s')
plt.title("Precision-Recall Curve", pad=15)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.grid(color='#334155', linestyle='--', linewidth=0.5)
plt.savefig('static/images/precision_recall_curve.png', bbox_inches='tight', transparent=True, dpi=300)
plt.close()

# 5. Feature Importance Bar Chart
print("Generating Feature Importance Bar Chart...")
feature_importances = plot_data["feature_importances"]
feature_cols = plot_data["feature_cols"]

indices = np.argsort(feature_importances)
sorted_features = [feature_cols[i] for i in indices][-10:]
sorted_importances = [feature_importances[i] for i in indices][-10:]

plt.figure(figsize=(10, 6))
plt.barh(sorted_features, sorted_importances, color='#94a3b8')
plt.title("Native Feature Importances (Gini Info Gain)", pad=15)
plt.xlabel("Average Mathematical Importance Ratio")
plt.tight_layout()
plt.savefig('static/images/feature_importance.png', bbox_inches='tight', transparent=True, dpi=300)
plt.close()

print("All visual graphs generated successfully using native algorithm payloads!")
