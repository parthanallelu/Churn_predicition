import csv
import json
import random

class CustomLabelEncoder:
    """
    Encodes categorical string labels into integer indices.

    Assigns a unique integer to each unique value, sorted alphabetically
    so encoding is deterministic. Unknown values at transform time fall
    back to index 0.

    Example:
        enc = CustomLabelEncoder()
        enc.fit_transform(['cat', 'dog', 'cat'])  # -> [0, 1, 0]
    """

    def __init__(self):
        self.classes_ = {}

    def fit(self, data):
        unique_vals = sorted(list(set(data)))
        self.classes_ = {val: idx for idx, val in enumerate(unique_vals)}
        return self

    def transform(self, data):
        return [self.classes_.get(val, 0) for val in data]

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

class Node:
    """
    A single node in a decision tree.

    Decision nodes store a split condition (feature_index + threshold) and
    pointers to left/right children. Leaf nodes store a class value and have
    no children.
    """

    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index  # which feature to split on
        self.threshold = threshold          # split value: left <= threshold, right > threshold
        self.left = left                    # left child Node
        self.right = right                  # right child Node
        self.value = value                  # class label (leaf nodes only)

class CustomDecisionTree:
    """
    Binary decision tree classifier using Gini impurity for node splitting.

    Implements the CART algorithm from scratch — no sklearn dependency.
    Recursively splits nodes by maximising information gain until
    max_depth or min_samples_split stopping criteria are met.

    Args:
        min_samples_split (int): Minimum samples required at a node to attempt a split. Default: 2.
        max_depth (int): Maximum depth of the tree. Deeper = more overfit. Default: 10.

    Attributes:
        root (Node): Root node of the fitted tree.
        feature_importances (list[float]): Per-feature importance scores after fitting.

    Example:
        tree = CustomDecisionTree(max_depth=8)
        tree.fit(X_train, y_train)
        preds = tree.predict(X_test)
    """

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
        
        # Pre-calculate parent gini to save time
        y_parent = [row[-1] for row in dataset]
        parent_gini = self._gini(y_parent)
        
        for feature_index in range(num_features):
            feature_values = [row[feature_index] for row in dataset]
            possible_thresholds = set(feature_values)
            
            if len(possible_thresholds) > 15: # Reduced from 20 for even more speed
                possible_thresholds = random.sample(list(possible_thresholds), 15)
                
            for threshold in possible_thresholds:
                # Fast split stats without creating full list datasets
                left_y = []
                right_y = []
                for row in dataset:
                    if row[feature_index] <= threshold:
                        left_y.append(row[-1])
                    else:
                        right_y.append(row[-1])
                
                if len(left_y) > 0 and len(right_y) > 0:
                    current_info_gain = self._information_gain_from_stats(y_parent, left_y, right_y, parent_gini)
                    
                    if current_info_gain > max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["info_gain"] = current_info_gain
                        max_info_gain = current_info_gain
        
        # Performance: Only do the actual data-split ONCE for the winner
        if "feature_index" in best_split:
            f_idx = best_split["feature_index"]
            thresh = best_split["threshold"]
            best_split["dataset_left"] = [row for row in dataset if row[f_idx] <= thresh]
            best_split["dataset_right"] = [row for row in dataset if row[f_idx] > thresh]
                        
        return best_split
        
    def _information_gain_from_stats(self, parent, l_child, r_child, parent_gini):
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        gain = parent_gini - (weight_l * self._gini(l_child) + weight_r * self._gini(r_child))
        return gain
        
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
    """
    Bagging (Bootstrap Aggregating) ensemble of CustomDecisionTree classifiers.

    Trains n_estimators trees, each on a balanced bootstrap sample of the
    training data (equal positive and negative samples). Predictions are
    made by majority vote; probabilities by averaging vote fractions.

    Args:
        n_estimators (int): Number of decision trees in the ensemble. Default: 10.

    Attributes:
        trees (list[CustomDecisionTree]): The fitted decision trees.

    Example:
        clf = CustomBaggingClassifier(n_estimators=30)
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)
        preds = clf.predict(X_test)
    """

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
        pos_idx = [i for i, label in enumerate(y) if label == 1]
        neg_idx = [i for i, label in enumerate(y) if label == 0]
        
        for i in range(self.n_estimators):
            print(f"  Training tree {i+1}/{self.n_estimators}...")
            # Balanced Bootstrap sample
            X_boot = []
            y_boot = []
            
            # Optimization: Cap sample size per tree to 2000 for speed
            # 1000 positive, 1000 negative
            max_per_class = 1000
            n_pos = min(len(pos_idx), max_per_class)
            n_neg = min(len(neg_idx), max_per_class)
            
            for _ in range(n_pos):
                p_i = random.choice(pos_idx)
                X_boot.append(X[p_i])
                y_boot.append(y[p_i])
            for _ in range(n_neg):
                n_i = random.choice(neg_idx)
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

def calculate_metrics(y_true, y_pred):
    """
    Compute binary classification metrics from true and predicted labels.

    Args:
        y_true (list[int]): Ground-truth labels (0 or 1).
        y_pred (list[int]): Predicted labels (0 or 1).

    Returns:
        dict with keys: accuracy, precision, recall, f1_score, confusion_matrix.
        confusion_matrix is [[TN, FP], [FN, TP]].
    """
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
    
    filename = 'cell2celltrain.csv'
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        
        # Identify columns
        feature_cols = [c for c in header if c not in ["CustomerID", "Churn"]]
        churn_idx = header.index("Churn")
        
        # Store for imputation
        # col_vals[idx] = [valid_values]
        col_vals = {c: [] for c in feature_cols}
        
        raw_rows = []
        for row in reader:
            if not row or len(row) < len(header): continue
            
            # Map Churn (Yes=1, No=0)
            y_val = 1 if row[churn_idx].strip().lower() == 'yes' else 0
            
            features_dict = {c: row[header.index(c)] for c in feature_cols}
            raw_rows.append((features_dict, y_val))
            
            # Collect valid values for imputation
            for c in feature_cols:
                val = features_dict[c]
                if val and val.strip().lower() not in ["", "na", "nan", "null"]:
                    col_vals[c].append(val)
                    
    print(f"Read {len(raw_rows)} rows. Calculating imputation stats...")
    
    # Calculate imputation values (Mean for numeric, Mode for strings)
    impute_map = {}
    is_numeric = {}
    
    for c in feature_cols:
        vals = col_vals[c]
        if not vals:
            impute_map[c] = 0
            is_numeric[c] = True
            continue
            
        # Try to see if it's numeric
        try:
            numeric_vals = [float(v) for v in vals]
            impute_map[c] = sum(numeric_vals) / len(numeric_vals)
            is_numeric[c] = True
        except ValueError:
            # Categorical: use Mode
            from collections import Counter
            counts = Counter(vals)
            impute_map[c] = counts.most_common(1)[0][0]
            is_numeric[c] = False

    # Process rows: Impute and Convert
    print("Processing features (imputation & casting)...")
    for feat_dict, y_val in raw_rows:
        row_vec = []
        for c in feature_cols:
            val = feat_dict[c]
            # Check for missing
            if not val or val.strip().lower() in ["", "na", "nan", "null"]:
                val = impute_map[c]
            
            if is_numeric[c]:
                row_vec.append(float(val))
            else:
                row_vec.append(str(val))
        
        X_raw.append(row_vec)
        y_raw.append(y_val)
            
    # Encode categorical columns
    print("Encoding categorical features...")
    encoders = {}
    for col_idx, col_name in enumerate(feature_cols):
        if not is_numeric[col_name]:
            encoder = CustomLabelEncoder()
            col_data = [row[col_idx] for row in X_raw]
            encoded_vals = encoder.fit_transform(col_data)
            for r_idx, row in enumerate(X_raw):
                row[col_idx] = encoded_vals[r_idx]
            encoders[col_name] = encoder
            
    print(f"Final dataset: {len(X_raw)} samples, {len(feature_cols)} features.")
    
    # Shuffle sync *BEFORE* splitting
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
    # Using 30 trees, max_depth 8 as per plan
    bagging_model = CustomBaggingClassifier(n_estimators=30)
    bagging_model.fit(X_train, y_train)
    
    # Metrics
    print("Calculating metrics...")
    preds = bagging_model.predict(X_test)
    metrics = calculate_metrics(y_test, preds)
    print(f"Results: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}")
    
    print("Saving model artifacts...")
    with open('custom_model.pkl', 'wb') as f:
        pickle.dump(bagging_model, f)
    with open('custom_encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)
    with open('custom_features.json', 'w') as f:
        json.dump(feature_cols, f)
    with open('custom_metrics.json', 'w') as f:
        json.dump(metrics, f)
        
    # Generate extra mathematical chart data
    print("Generating analytics dashboard data...")
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
    print("Training process complete.")
