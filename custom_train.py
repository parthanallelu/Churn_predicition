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

    def fit(self, X, y):
        # Merge X and y for easy splitting
        dataset = [x + [y_val] for x, y_val in zip(X, y)]
        self.root = self._build_tree(dataset, depth=0)
        
    def _build_tree(self, dataset, depth):
        X, y = [row[:-1] for row in dataset], [row[-1] for row in dataset]
        num_samples, num_features = len(X), len(X[0])

        if num_samples >= self.min_samples_split and depth <= self.max_depth:
            # Find best split
            best_split = self._get_best_split(dataset, num_samples, num_features)
            
            if best_split.get("info_gain", 0) > 0:
                left_subtree = self._build_tree(best_split["dataset_left"], depth + 1)
                right_subtree = self._build_tree(best_split["dataset_right"], depth + 1)
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
        
    def fit(self, X, y):
        print(f"Training {self.n_estimators} Custom Decision Trees...")
        n_samples = len(X)
        for i in range(self.n_estimators):
            print(f"  Training tree {i+1}/{self.n_estimators}...")
            # Bootstrap sample
            X_boot = []
            y_boot = []
            for _ in range(n_samples):
                idx = random.randint(0, n_samples - 1)
                X_boot.append(X[idx])
                y_boot.append(y[idx])
                
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
    
    with open('WA_Fn-UseC_-Telco-Customer-Churn.csv', 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        
        # customerID is col 0, Churn is the last col (20). 
        # Identify columns
        feature_cols = header[1:-1]
        
        total_charges_idx = header.index("TotalCharges")
        churn_idx = header.index("Churn")
        
        # Store for mean imputation
        total_charges_vals = []
        
        for row in reader:
            features = row[1:-1]
            y_val = 1 if row[churn_idx] == "Yes" else 0
            
            # Clean TotalCharges
            tc = row[total_charges_idx]
            try:
                tc_float = float(tc)
                total_charges_vals.append(tc_float)
            except ValueError:
                features[total_charges_idx - 1] = None # Will impute later
                
            X_raw.append(features)
            y_raw.append(y_val)
            
    # Impute missing TotalCharges with mean
    mean_tc = sum(total_charges_vals) / len(total_charges_vals) if total_charges_vals else 0
    for row in X_raw:
        if row[total_charges_idx - 1] is None:
            row[total_charges_idx - 1] = mean_tc
        else:
            row[total_charges_idx - 1] = float(row[total_charges_idx - 1])
            
    # Also numeric cast for tenure (idx 4 if customerId dropped), MonthlyCharges (idx 17)
    tenure_idx = feature_cols.index("tenure")
    monthly_idx = feature_cols.index("MonthlyCharges")
    senior_idx = feature_cols.index("SeniorCitizen")
    
    for row in X_raw:
        row[tenure_idx] = float(row[tenure_idx])
        row[monthly_idx] = float(row[monthly_idx])
        row[senior_idx] = float(row[senior_idx])
        
    # Encode categorical columns statically
    print("Encoding matrices natively...")
    encoders = {}
    for col_idx, col_name in enumerate(feature_cols):
        if col_idx not in [tenure_idx, monthly_idx, total_charges_idx - 1, senior_idx]:
            # This is a categorical string column
            encoder = CustomLabelEncoder()
            col_data = [row[col_idx] for row in X_raw]
            encoded_vals = encoder.fit_transform(col_data)
            for r_idx, row in enumerate(X_raw):
                row[col_idx] = encoded_vals[r_idx]
            encoders[col_name] = encoder
            
    print(f"Engineered {len(X_raw)} rows natively.")
    
    # Train/Test Split (80/20)
    split_idx = int(len(X_raw) * 0.8)
    
    # Shuffle sync
    combined = list(zip(X_raw, y_raw))
    random.seed(42)
    random.shuffle(combined)
    X_raw, y_raw = zip(*combined)
    X_raw, y_raw = list(X_raw), list(y_raw)
    
    X_train, y_train = X_raw[:split_idx], y_raw[:split_idx]
    X_test, y_test = X_raw[split_idx:], y_raw[split_idx:]
    
    # Train
    bagging_model = CustomBaggingClassifier(n_estimators=10) # Reduced count to speed up pure python testing
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
