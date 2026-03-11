# Project Codebase Explanation

This document provides a breakdown of every file present in the Customer Churn Prediction project. It explains the architectural layout, script responsibilities, and how all the files connect to build the complete Machine Learning web application pipeline from scratch without relying heavily on external libraries like `pandas` or `scikit-learn`.

---

## 1. Machine Learning Engine (Backend)

These are the core algorithmic files that train the mathematical model and pre-calculate required metrics.

### `custom_train.py`

**The absolute core of the project's logic.** This script is executed first and replaces the traditional Jupyter Notebook. It performs all the heavy lifting using only native Python modules (`csv`, `math`, `random`).

- **Reads the Dataset:** It manually parses the `WA_Fn-UseC_-Telco-Customer-Churn.csv` file row-by-row into memory.
- **Data Cleaning:** It iterates through the dataset to locate missing/corrupt values in the `TotalCharges` column, calculates the mathematical mean of valid entries, and automatically imputes (fixes) the empty gaps.
- **Custom Classes Built From Scratch:**
  - `CustomLabelEncoder`: A class that assigns integer IDs to Categorical strings (e.g., Mapping `Month-to-month -> 0`).
  - `Node` & `CustomDecisionTree`: Recursive algorithmic classes that build binary decision nodes based on calculated Gini Impurity scores.
  - `CustomBaggingClassifier`: An ensemble wrapper that builds Random bootstrapped arrays (sampling with replacement) and trains 10 independent versions of the Decision Trees to stabilize predictions.
- **Execution & Metric Calculation:** The script shards the data into 80% Training and 20% Testing sets. It trains the Bagging Model and manually calculates mathematical evaluation metrics array (Accuracy, Precision, Recall, Confusion Matrix dimensions).
- **Serialization:** Finally, it uses Python's built-in `pickle` library to save the entire trained mathematical framework to disk, avoiding the need to retrain nodes every time the evaluator opens the web application.

## 2. Generated Model Artifacts

These files are the serialized (saved) outputs generated directly by running `custom_train.py`. The Flask server relies exclusively on these files to serve rapid predictions without doing retraining.

- **`custom_model.pkl`**: A binary pickle file containing the fully trained, multi-thousand node `CustomBaggingClassifier` object.
- **`custom_encoders.pkl`**: A saved dictionary mapping all text-based fields (Gender, Contract type) to the exact integers the ML model was trained on.
- **`custom_features.json`**: A strict chronological array of the dataset headers. This ensures the Flask server feeds user inputs into the model exactly in the same column order standard as the CSV file.
- **`custom_metrics.json`**: A dictionary containing the exact mathematical validation results (Accuracy percentage, False-Positive rates) calculated natively on the 20% validation split.

## 3. Web Service & Presentation Layer

These files handle translating raw algorithmic arrays into a stunning user-friendly website.

### `app.py`

**The Flask API Server.** This is the bridge file that connects the ML logic to the browser interface.

- **Initialization:** When started via `python app.py`, the script immediately deserializes (loads) `custom_model.pkl`, `custom_encoders.pkl`, etc., directly into server memory.
- **The `/` Route:** Serves the `index.html` file to the evaluator. It intercepts `custom_metrics.json` and passes these mathematical scores directly into the HTML templating engine so they are rendered beautifully on screen.
- **The `/about` and `/docs` Routes:** Serves the `about.html` multi-page view, which details the problem statement natively. The `/docs/<filename>` API securely allows `marked.js` to fetch and render specific project markdown files securely right inside the browser window.
- **The `/predict` Engine:** A `POST` endpoint that listens for dictionary inputs from the web UI. It receives a JSON payload like `{"Contract": "Month-to-month"}`, iterates the value through the loaded `encoders` dictionary to get the integer mapping, maps it mathematically into a perfect array shape, and runs it through the loaded ML Model. The resulting probability is packaged as JSON and fired back down to the browser.

### `generate_graphs.py`

A supplementary script utilizing `matplotlib` and `seaborn`. It generates visual charts directly analyzing the `.csv` data (Churn distributions, Box Plots, Feature Correlations). These are saved as static images styled to blend in beautifully with the browser's dark mode aesthetic.

### `requirements.txt`

A highly minimal dependency tracking file. Because the project leverages Custom Python classes for algorithms, the only dependencies are those managing Web Routing (`Flask`), plotting (`matplotlib`), and their sub-requirements.

---

## 4. The Frontend Interface (`/templates` and `/static`)

This folder powers the evaluator's interaction window without node.js packaging overhead.

### `templates/index.html` & `templates/about.html`

- The raw HTML structures containing the formal presentation. `index.html` leverages **Jinja2** (e.g., `{{ metrics.accuracy }}`) templating to dynamically inject Machine Learning array analytics from `app.py`. `about.html` contains the mathematical background and utilizes CDN scripts to asynchronously fetch and render external `.md` evaluation reports seamlessly inline.

### `static/style.css`

- A premium CSS file utilizing Glassmorphism design principles containing formal monochromatic palettes (slate/silver), custom animated `@keyframes`, and responsive flexible grids. It shapes the metrics dashboards, navigation bars, and embedded markdown viewers into a cohesive enterprise-grade user experience.

### `static/script.js`

- The active listener engine on the client side. When the evaluator clicks the "Predict Churn Risk" button, this script instantly bundles all form values into a JSON dictionary payload and fires an asynchronous `fetch()` request back up to the `app.py` `/predict` API endpoint. Upon receiving the mathematical outcome, the script parses the probability array, changes UI colors based on risk severity, and animates the probability integers on screen beautifully.
