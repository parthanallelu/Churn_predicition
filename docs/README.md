# Customer Churn Prediction: Final Project Report

## 1. Problem Statement (PS)

In the highly competitive telecommunications sector, acquiring new customers is significantly more expensive than retaining existing ones. Companies frequently lose revenue due to "customer churn"—when subscribers abruptly cancel their services in favor of competitors. The problem lies in identifying exactly _which_ active customers are at a high risk of churning before they actually leave, allowing the business to proactively offer targeted retention incentives.

## 2. Objectives

This project aims to solve the customer churn problem by developing a robust predictive machine learning web application.

- **Primary Objective:** To build a Machine Learning model completely from scratch (via pure mathematical Python equations without heavy libraries like `scikit-learn` or `pandas`) that can predict whether a Telco customer will churn based on historical demographic and billing data.
- **Secondary Objective (Accuracy):** To stabilize the high-variance nature of native Decision Trees by implementing a **Bagging (Bootstrap Aggregating)** ensemble architecture natively.
- **Tertiary Objective (Deployment):** To construct a premium, interactive formal "Glassmorphism" web application with a two-page architecture. The dashboard allows evaluators to tweak hypothetical customer profiles and view predictions in real-time, while a dedicated `/about` page natively renders architectural `.md` documentations dynamically via `marked.js`.

## 3. Introduction

Customer Churn Prediction is a binary classification problem (Predicting Yes vs. No). By analyzing a publicly available Telco dataset containing over 7,000 customer rows covering `Tenure`, `Monthly Charges`, `Internet Services`, and `Payment Methods`, algorithms can mathematically correlate patterns in canceling users. This project bridges the gap between raw data science and software engineering by serving custom-written algorithmic predictive logic through a modern web-based Flask REST API.

## 4. Literature Review

The shift towards predictive analytics in customer retention has been heavily studied:

- **Decision Trees (CART):** Described fundamentally by Breiman et al. (1984), Classification and Regression Trees are powerful non-parametric algorithms predicting values by learning simple decision rules. However, they are mathematically prone to overfitting (memorizing the training dataset).
- **Ensemble Learning (Bagging):** To resolve overfitting, Breiman (1996) introduced _Bagging Predictors_, wherein multiple versions of a predictor are trained on bootstrap replicates of the learning set, and their outputs are averaged to form a highly stabilized, generalized prediction.
- **Production Constraints:** While modern frameworks like Scikit-Learn abstract these mathematical complexes beautifully, they introduce substantial dependency weight for simple web deployments. Writing the underlying Gini-Impurity logic from scratch in native algorithmic structures showcases a deeper fundamental understanding of Machine Learning math.

## 5. Architectural Diagrams

```mermaid
graph TD
    A[Raw Dataset (CSV)] -->|Read row by row| B(Native Preprocessing)
    B -->|Mean Imputation & Label Encoding| C{Train/Test Vector Split}

    C -->|80% Training Data| D[Custom Bagging Classifier]
    D -->|Random Bootstrap Loop| E[Train 10x Decision Trees]
    E -->|Gini Node Mathematics| F[(custom_model.pkl)]

    C -->|20% Validation Data| G[Evaluation Metrics engine]
    G --> H[(custom_metrics.json)]

    I[User Website Input] -->|JSON POST| J[Flask API /predict]
    F --> J
    H -->|Render directly| K[index.html Output]
    J -->|Prediction Probability| K
```

## 6. Pseudo Code / Algorithm

Below is the conceptual algorithmic flow of the primary engine (`custom_train.py`):

```pseudo
BEGIN
  LOAD dataset from CSV into List mapping
  STORE column headers

  // PREPROCESSING
  FOR EACH row IN dataset:
      IF TotalCharges is empty:
          SET TotalCharges = MEAN(all_valid_TotalCharges)
      ENCODE Categorical Strings to Integers (e.g. Month-to-month = 0)

  // SPLITTING
  SHUFFLE dataset
  X_train, y_train = FIRST 80% OF dataset
  X_test, y_test = REMAINING 20% OF dataset

  // MODEL DEFINITION (Bagging Wrapper)
  FUNCTION Bagging_Train(X, y, num_trees):
      trees = []
      FOR i in range(num_trees):
          X_boot, y_boot = RANDOM_SAMPLE_WITH_REPLACEMENT(X, y)
          tree = DecisionTree_Train(X_boot, y_boot)
          trees.APPEND(tree)
      RETURN trees

  // MODEL ALGORITHM (Decision Tree recursive branching)
  FUNCTION DecisionTree_Train(data, current_depth):
      IF max_depth REACHED OR data is pure:
          RETURN Leaf_Node(majority_vote)

      best_split = FIND_HIGHEST_GINI_GAIN(data)

      left_branch = DecisionTree_Train(best_split.left_data, current_depth + 1)
      right_branch = DecisionTree_Train(best_split.right_data, current_depth + 1)

      RETURN Decision_Node(best_split.feature, best_split.threshold, left_branch, right_branch)

  // PREDICTION
  FUNCTION Predict(input_vector):
      votes = []
      FOR tree in trained_trees:
          votes.APPEND(tree.predict(input_vector))
      RETURN MAJORITY(votes), PROBABILITY(votes)

  // EXECUTION
  model = Bagging_Train(X_train, y_train, 10)
  metrics = Calculate_Confusion_Matrix(model.predict(X_test), y_test)

  SAVE model TO "custom_model.pkl"
  SAVE metrics TO "custom_metrics.json"
END
```

## 7. Output Results

The implementation generated a highly accurate mathematical classifier with zero dependence on the pandas or sklearn library stack. The metrics obtained via the `custom_metrics.json` validation split were evaluated as follows:

- **Confusion Matrix:** Evaluated exactly how many False Positives mapped vs True Positives without relying on external algorithms.
- **Precision:** Guaranteed that mathematical alerts regarding leaving-customers were highly targeted.
- **Web Portal Processing:** Natively casting dictionaries through the custom Label Encoder array mapping means predictions render beautifully through the Flask engine dynamically across `index.html`.

## 8. Conclusion

This project successfully achieved its goal of building an interactive, end-to-end Machine Learning web application utilizing natively written classification algorithms.

By systematically stripping away "black-box" libraries (like Scikit-Learn classification algorithms and Pandas dataset handlers) in favor of custom-written arrays, Gini Index calculus, and Bootstrap subset loops, the pipeline demonstrates a profound functional understanding of Machine Learning math.

Deploying these lightweight binary files natively via Flask into a premium formal Glassmorphism HTML dashboard proves that complex, production-ready predictive analytical apps can be built from raw fundamentals. With integrated dynamic Markdown-viewers and cohesive native data-visualizations, the project is a fully-featured, easily explainable, performant, and highly reliable Machine Learning web portal.
