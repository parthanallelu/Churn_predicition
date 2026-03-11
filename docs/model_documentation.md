# Customer Churn Prediction: Machine Learning Model Documentation

This document provides a comprehensive overview of the machine learning pipeline, algorithms, methodologies, and evaluation criteria utilized in the Customer Churn Prediction project.

---

## 1. Project Overview & Objective

The primary objective of this project is to develop a predictive model that can identify whether a telecommunications customer is likely to "churn" (cancel their service). Retaining existing customers is significantly more cost-effective than acquiring new ones, making this predictive analysis crucial for targeted retention programs.

## 2. Theoretical Foundations

### 2.1 Customer Churn Prediction

Churn prediction is a classic **Binary Classification** problem. The target variable is `Churn`, which has two possible outcomes:

- **Yes (1):** The customer left within the last month.
- **No (0):** The customer remained subscribed.

By analyzing historical data featuring demographic info, service subscriptions, and billing behavior, the model learns the patterns associated with customers who cancel.

### 2.2 Decision Trees (The Base Estimator)

A Decision Tree is a non-parametric supervised learning algorithm. It works by splitting the data into subsets based on the value of input features.

- **How it works:** It asks a series of "if-else" questions about individual features (e.g., "Is Tenure < 12 months?", "Is the contract Month-to-Month?").
- **Pros:** Highly interpretable, handles non-linear relationships well, and requires minimal data scaling.
- **Cons:** Highly prone to **overfitting** (memorizing the training data and failing to generalize to new, unseen customers).

### 2.3 Bagging (Bootstrap Aggregating)

To combat the high variance and overfitting tendencies of a single Decision Tree, this project utilizes an Ensemble Learning technique called **Bagging**.

- **Bootstrap Sampling:** The algorithm creates multiple subsets (in this project, `n_estimators=50`) of the original training data by sampling _with replacement_.
- **Parallel Training:** A separate, independent Decision Tree is trained on each of these 50 bootstrapped subsets.
- **Aggregating (Voting):** When making a prediction for a new customer, all 50 trees cast a "vote" (Yes or No). The final prediction is determined by majority voting, while the probability is the ratio of votes (e.g., 40 trees say Yes, 10 say No = 80% Churn Probability).
- **Advantages:** Dramatically reduces model variance, improves accuracy, and provides a much more robust, generalized prediction model than relying on any single tree.

---

## 3. The Implementation Pipeline (Custom Python Engineering)

Following a strict requirement to minimize external dependencies, the typical data science stack (`pandas`, `scikit-learn`) was entirely omitted. The machine learning pipeline was constructed utilizing pure Python arrays, math functions, and native data structures.

### 3.1 Native Data Preprocessing

Raw data is parsed manually using Python's built-in `csv` module. The following preprocessing steps were undertaken linearly:

1.  **Feature Selection:** The `customerID` column was explicitly ignored during parsing. While unique to each user, it carries no predictive mathematical weight and can confuse the model.
2.  **Handling Missing / Corrupted Data:** Missing values (often hidden as whitespaces in the `TotalCharges` column for new customers) were detected dynamically using `try/except float()` blocks. The custom pipeline tracks all valid floats, computes their mathematical mean natively, and iterates back over the dataset to Impute (fill) the faulty entries with this calculated average without dropping array rows.
3.  **Custom Label Encoding:** Machine learning algorithms fundamentally rely on numerical mathematics. Since columns like `Gender`, `Contract`, and `PaymentMethod` contain text string values, we built a `CustomLabelEncoder` class. This class iterates through unique strings within an array column, generating a sequential `{string: integer}` dictionary lookup map (e.g., Month-to-month = 0, One year = 1, Two year = 2). These dictionaries are serialized so web inputs can be passed through the exact same mapping rules.

### 3.2 Model Training (Algorithmic Construction)

1.  **Data Splitting:** A native 80/20 data split was constructed by mapping the 2D feature matrices to the `y` target variables array, seeding a `random.shuffle()` distribution, and slicing the resulting arrays.
2.  **Initialization:** A `CustomDecisionTree` class was written utilizing Gini index heuristics to execute binary node branching down to a `max_depth` limit.
3.  **Ensemble Bootstrapping:** A `CustomBaggingClassifier` wrapper manages the training loops. During `bagging_model.fit()`, it uses `random.randint` to execute _sampling with replacement_ (Bootstrapping), generating datasets of equal size to train 10 independent versions of the custom Decision Tree recursively.
4.  **Serialization:** Using Python's built-in `pickle` and `json` libraries, the fully trained Bagging object, alongside the dictionary of Label Encoders and the required feature schema, is saved locally (`custom_model.pkl`, `custom_encoders.pkl`, `custom_features.json`). This guarantees the backend web application executes inferences instantly natively without needing to parse the original 7000 rows again.

### 3.3 Prediction Generation

When the Flask API receives a `POST` request with a new customer's JSON characteristics from the web UI:

1.  The dictionary is parsed iteratively and aligned perfectly to the expected feature schema length.
2.  Categorical string fields are dynamically looked up against the deserialized `encoder.classes_` dictionary.
3.  `model.predict_proba()` executes. The Custom Bagging classifier runs the integer-mapped array through all 10 underlying decision trees individually recursively down to their leaf nodes.
4.  The output ratio provides the consensus confidence percentage across the ensemble for Class 0 (No Churn) and Class 1 (Churn), which is returned identically sized to the frontend script.

---

## 4. Evaluation Metrics

Evaluating the native test dataset (20% holdout) reveals how the custom algorithms generalize on customers it hasn't mapped before. The custom metrics mathematical formula outcomes are stored inside `custom_metrics.json` and rendered on the frontend dashboard.

### 4.1 Confusion Matrix

The confusion matrix is the foundational table that breaks down all prediction behaviors across 4 quadrants. Our custom `calculate_metrics()` loops over the raw predictions counting the conditional agreements:

- **True Positives (TP):** Model predicted _Churn_, and the customer _actually churned_. (Successful detection).
- **True Negatives (TN):** Model predicted _Retained_, and the customer _remained_. (Successful detection).
- **False Positives (FP) [Type I Error]:** Model predicted _Churn_, but the customer _remained_. (Model was too aggressive/pessimistic).
- **False Negatives (FN) [Type II Error]:** Model predicted _Retained_, but the customer _actually churned_. (The most dangerous error for a business—a missed opportunity for retention).

### 4.2 Accuracy

- **Definition:** The overall percentage of correct predictions out of all predictions made.
- **Formula:** `(TP + TN) / (TP + TN + FP + FN)`
- **Context:** While widely used, high Accuracy can be misleading in imbalanced datasets (e.g., if only 10% of total customers churn, a "dumb" model that always guesses "No" achieves 90% accuracy but is entirely useless).

### 4.3 Precision

- **Definition:** Out of all the customers the model _flagged_ as "Likely to Churn," what percentage actually did?
- **Formula:** `TP / (TP + FP)`
- **Context:** Measures the "quality" or exactness of the positive alerts. High precision means when the system flags a user for a retention program, it's very rarely a false alarm (low False Positives).

### 4.4 Recall (Sensitivity)

- **Definition:** Out of all the customers who _actually churned_ in reality, what percentage did the model successfully find?
- **Formula:** `TP / (TP + FN)`
- **Context:** Measures the "quantity" or completeness of detections. High recall means the algorithm is catching almost all the leaving customers, though it might suffer from False Positives to achieve it.

### 4.5 F1-Score

- **Definition:** The Harmonic Mean of Precision and Recall.
- **Formula:** `2 * (Precision * Recall) / (Precision + Recall)`
- **Context:** It provides a single, balanced metric ensuring both Precision and Recall are healthy. The harmonic mean punishes the score heavily if either Precision or Recall are critically low.

---

## 5. Software Stack & Libraries

To comply with the strict zero-external-dependency rule (where possible), heavy libraries like Scikit-Learn and Pandas were aggressively omitted.

### Pure Python (Native Architecture)

- `csv`: Relied on heavily for manually parsing multidimensional data structures into lists and casting strings.
- `math / json`: Used extensively to calculate algorithm formulas and pass dynamic dictionary payloads to the API.
- `random`: Powers the array slicing, Train/Test 80/20 list shuffling, and importantly, the Bagging Bootstrap subsampling mathematical engine.
- `pickle`: Serializes complex class instance objects (like our fully trained `CustomBaggingClassifier` objects spanning thousands of nodes) straight to binary files for rapid server restarts.

### Presentation Output

- **Flask:** Micro web framework running Python endpoints (`@app.route`). It bridges the gap between the internal ML environments and the vanilla Javascript API calls (`jsonify`, `request.json`).
- **Jinja (`render_template`):** Used to inject backend analytical contexts (the JSON metrics object and image file paths) dynamically into the static `index.html`.
- **Vanilla HTML/CSS/JS**: Creates the responsive formal Glassmorphism interface. The platform spans a real-time prediction application (`index.html`) alongside a dedicated architecture analysis viewer (`about.html`) that parses document frameworks seamlessly via `marked.js` APIs.
