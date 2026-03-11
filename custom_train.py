import csv
import json
import random

class CustomLabelEncoder:
    def __init__(self):
        self.classes_ = {}
        
    def fit(self, data):
        unique_vals = sorted(list(set(data)))
        self.classes_ = {val: idx for idx, val in enumerate(unique_vals)}
        return self
        
    def transform(self, data):
        return [self.classes_.get(val, 0) for val in data] # Fallback to 0 if unknown 
        
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        # Decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        # Leaf node
        self.value = value

class CustomDecisionTree:
    def __init__(self, min_samples_split=2, max_depth=10):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None
        self.feature_importances = None

    def fit(self, X, y):
        self.feature_importances = [0.0] * len(X[0])
        # Merge X and y for easy splitting
        dataset = [x + [y_val] for x, y_val in zip(X, y)]
        self.root = self._build_tree(dataset, depth=0, total_samples=len(X))
        
    def _build_tree(self, dataset, depth, total_samples):
        X, y = [row[:-1] for row in dataset], [row[-1] for row in dataset]
        num_samples, num_features = len(X), len(X[0])

        if num_samples >= self.min_samples_split and depth <= self.max_depth:
            # Find best split
            best_split = self._get_best_split(dataset, num_samples, num_features)
            
            if best_split.get("info_gain", 0) > 0:
                self.feature_importances[best_split["feature_index"]] += (best_split["info_gain"] * num_samples / total_samples)
                left_subtree = self._build_tree(best_split["dataset_left"], depth + 1, total_samples)
                right_subtree = self._build_tree(best_split["dataset_right"], depth + 1, total_samples)
                return Node(best_split["feature_index"], best_split["threshold"], left_subtree, right_subtree)
                
        # Leaf Node
        leaf_value = self._calculate_leaf_value(y)
        return Node(value=leaf_value)
        
    def _get_best_split(self, dataset, num_samples, num_features):
        best_split = {}
        max_info_gain = -float("inf")
        
        for feature_index in range(num_features):
            feature_values = [row[feature_index] for row in dataset]
            possible_thresholds = set(feature_values)
            
            # Subsample thresholds if too many (Optimization for continuous vars)
            if len(possible_thresholds) > 20: 
                possible_thresholds = random.sample(list(possible_thresholds), 20)
                
            for threshold in possible_thresholds:
                dataset_left, dataset_right = self._split(dataset, feature_index, threshold)
                
                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    y, left_y, right_y = [row[-1] for row in dataset], [row[-1] for row in dataset_left], [row[-1] for row in dataset_right]
                    current_info_gain = self._information_gain(y, left_y, right_y)
                    
                    if current_info_gain > max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = current_info_gain
                        max_info_gain = current_info_gain
                        
        return best_split
        
    def _split(self, dataset, feature_index, threshold):
        dataset_left = [row for row in dataset if row[feature_index] <= threshold]
        dataset_right = [row for row in dataset if row[feature_index] > threshold]
        return dataset_left, dataset_right
        
    def _information_gain(self, parent, l_child, r_child):
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        gain = self._gini(parent) - (weight_l * self._gini(l_child) + weight_r * self._gini(r_child))
        return gain
        
    def _gini(self, y):
        # Calculate frequencies
        counts = {}
        for label in y:
            counts[label] = counts.get(label, 0) + 1
            
        impurity = 1
        for label in counts:
            prob_of_lbl = counts[label] / len(y)
            impurity -= prob_of_lbl ** 2
        return impurity

    def _calculate_leaf_value(self, Y):
        # Majority vote
        counts = {}
        for y in Y:
            counts[y] = counts.get(y, 0) + 1
        return max(counts, key=counts.get)
        
    def _predict_single(self, x, tree):
        if tree.value is not None:
            return tree.value
        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self._predict_single(x, tree.left)
        else:
            return self._predict_single(x, tree.right)

    def predict(self, X):
        return [self._predict_single(x, self.root) for x in X]

class CustomBaggingClassifier:
    def __init__(self, n_estimators=10):
        self.n_estimators = n_estimators
        self.trees = []
        
    @property
    def feature_importances_(self):
        if not self.trees: return []
        n_features = len(self.trees[0].feature_importances)
        avg = [0.0] * n_features
        for t in self.trees:
            for i in range(n_features):
                avg[i] += t.feature_importances[i]
        sum_imp = sum(avg)
        return [x/sum_imp for x in avg] if sum_imp > 0 else avg
        
    def fit(self, X, y):
        print(f"Training {self.n_estimators} Custom Decision Trees with balanced sampling...")
        # Get indices of positive and negative classes
        pos_idx = [i for i, label in enumerate(y) if label == 1]
        neg_idx = [i for i, label in enumerate(y) if label == 0]
        
        for i in range(self.n_estimators):
            print(f"  Training tree {i+1}/{self.n_estimators}...")
            # Balanced Bootstrap sample
            X_boot = []
            y_boot = []
            
            # Determine size based on the smaller class
            sample_size = min(len(pos_idx), len(neg_idx))
            
            for _ in range(sample_size):
                p_i = random.choice(pos_idx)
                n_i = random.choice(neg_idx)
                X_boot.append(X[p_i])
                y_boot.append(y[p_i])
                X_boot.append(X[n_i])
                y_boot.append(y[n_i])
                
            tree = CustomDecisionTree(max_depth=8)
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
            
    def predict_proba(self, X):
        # Probabilities
        all_tree_preds = [tree.predict(X) for tree in self.trees]
        # Transpose so rows are samples, cols are tree predictions
        sample_preds = list(map(list, zip(*all_tree_preds)))
        
        probas = []
        for preds in sample_preds:
            class_1_prob = sum(preds) / len(preds)
            probas.append([1 - class_1_prob, class_1_prob])
        return probas
        
    def predict(self, X):
        all_tree_preds = [tree.predict(X) for tree in self.trees]
        sample_preds = list(map(list, zip(*all_tree_preds)))
        
        final_preds = []
        for preds in sample_preds:
            # Majority vote
            ones = sum(preds)
            zeros = len(preds) - ones
            final_preds.append(1 if ones > zeros else 0)
        return final_preds

# Metrics calculations
def calculate_metrics(y_true, y_pred):
    tp = tn = fp = fn = 0
    for true, pred in zip(y_true, y_pred):
        if true == 1 and pred == 1: tp += 1
        elif true == 0 and pred == 0: tn += 1
        elif true == 0 and pred == 1: fp += 1
        elif true == 1 and pred == 0: fn += 1
        
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp+tn+fp+fn) > 0 else 0
    precision = tp / (tp + fp) if (tp+fp) > 0 else 0
    recall = tp / (tp + fn) if (tp+fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision+recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': [[tn, fp], [fn, tp]]
    }

if __name__ == "__main__":
    import pickle
    
    print("Loading data via native CSV parser...")
    X_raw = []
    y_raw = []
    header = []
    
    with open('customer_churn_dataset-testing-master.csv', 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        
        # Identify columns
        feature_cols = [c for c in header if c not in ["CustomerID", "Churn"]]
        
        total_charges_idx = header.index("Total Spend")
        churn_idx = header.index("Churn")
        
        # Store for mean imputation
        total_charges_vals = []
        
        for row in reader:
            # Clean empty/missing rows first
            if not row or not row[0]: continue
                
            features = [row[header.index(c)] for c in feature_cols]
            try:
                y_val = int(row[churn_idx])
            except ValueError:
                continue # Skip row if Target is blank
            
            # Clean Total Spend
            tc = row[total_charges_idx]
            try:
                tc_float = float(tc)
                total_charges_vals.append(tc_float)
            except ValueError:
                features[feature_cols.index("Total Spend")] = None # Will impute later
                
            X_raw.append(features)
            y_raw.append(y_val)
            
    # Impute missing Total Spend with mean
    mean_tc = sum(total_charges_vals) / len(total_charges_vals) if total_charges_vals else 0
    tc_feature_idx = feature_cols.index("Total Spend")
    for row in X_raw:
        if row[tc_feature_idx] is None:
            row[tc_feature_idx] = mean_tc
        else:
            row[tc_feature_idx] = float(row[tc_feature_idx])
            
    # Also numeric cast for Age, Tenure, Usage Frequency, Support Calls, Payment Delay, Last Interaction
    age_idx = feature_cols.index("Age")
    tenure_idx = feature_cols.index("Tenure")
    usage_idx = feature_cols.index("Usage Frequency")
    support_idx = feature_cols.index("Support Calls")
    payment_idx = feature_cols.index("Payment Delay")
    interaction_idx = feature_cols.index("Last Interaction")
    
    for row in X_raw:
        row[age_idx] = float(row[age_idx])
        row[tenure_idx] = float(row[tenure_idx])
        row[usage_idx] = float(row[usage_idx])
        row[support_idx] = float(row[support_idx])
        row[payment_idx] = float(row[payment_idx])
        row[interaction_idx] = float(row[interaction_idx])
        
    # Encode categorical columns statically
    print("Encoding matrices natively...")
    encoders = {}
    numeric_indices = [age_idx, tenure_idx, usage_idx, support_idx, payment_idx, interaction_idx, tc_feature_idx]
    for col_idx, col_name in enumerate(feature_cols):
        if col_idx not in numeric_indices:
            # This is a categorical string column
            encoder = CustomLabelEncoder()
            col_data = [row[col_idx] for row in X_raw]
            encoded_vals = encoder.fit_transform(col_data)
            for r_idx, row in enumerate(X_raw):
                row[col_idx] = encoded_vals[r_idx]
            encoders[col_name] = encoder
            
    print(f"Engineered {len(X_raw)} rows natively.")
    
    # Shuffle sync *BEFORE* splitting to prevent data leakage/ordered clumping
    combined = list(zip(X_raw, y_raw))
    random.seed(42)
    random.shuffle(combined)
    X_raw, y_raw = zip(*combined)
    X_raw, y_raw = list(X_raw), list(y_raw)
    
    # Train/Test Split (80/20)
    split_idx = int(len(X_raw) * 0.8)
    
    X_train, y_train = X_raw[:split_idx], y_raw[:split_idx]
    X_test, y_test = X_raw[split_idx:], y_raw[split_idx:]
    
    # Train
    bagging_model = CustomBaggingClassifier(n_estimators=30)
    bagging_model.fit(X_train, y_train)
    
    # Metrics
    print("Calculating native evaluation metrics...")
    preds = bagging_model.predict(X_test)
    metrics = calculate_metrics(y_test, preds)
    
    print("Exporting native binary data structures...")
    
    with open('custom_model.pkl', 'wb') as f:
        pickle.dump(bagging_model, f)
        
    with open('custom_encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)
        
    with open('custom_features.json', 'w') as f:
        json.dump(feature_cols, f)
        
    with open('custom_metrics.json', 'w') as f:
        json.dump(metrics, f)
        
    print("Native serialization complete -> custom_model.pkl, custom_encoders.pkl, custom_features.json, custom_metrics.json")
    
    # Generate extra mathematical chart data manually
    print("Generating algorithmic chart plotting data...")
    acc_vs_trees = []
    for i in range(1, bagging_model.n_estimators + 1):
        temp_model = CustomBaggingClassifier(n_estimators=i)
        temp_model.trees = bagging_model.trees[:i]
        preds_i = temp_model.predict(X_test)
        acc_i = calculate_metrics(y_test, preds_i)["accuracy"]
        acc_vs_trees.append(acc_i)
        
    probas = bagging_model.predict_proba(X_test)
    y_scores = [p[1] for p in probas]
    thresholds = [i/10.0 for i in range(11)]
    pr_curve = []
    for t in thresholds:
        preds_t = [1 if s >= t else 0 for s in y_scores]
        m = calculate_metrics(y_test, preds_t)
        pr_curve.append({"threshold": t, "precision": m["precision"], "recall": m["recall"]})
        
    plot_data = {
        "feature_importances": bagging_model.feature_importances_,
        "acc_vs_trees": acc_vs_trees,
        "pr_curve": pr_curve,
        "feature_cols": feature_cols,
        "confusion_matrix": metrics["confusion_matrix"]
    }
    with open('custom_plot_data.json', 'w') as f:
        json.dump(plot_data, f)
    print("Exported custom analytics logic -> custom_plot_data.json.")
