"""
Unit tests for the core ML components in custom_train.py.

Run with:  pytest tests/test_model.py -v
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from custom_train import (
    CustomLabelEncoder,
    CustomDecisionTree,
    CustomBaggingClassifier,
    calculate_metrics,
)


# ── CustomLabelEncoder ────────────────────────────────────────────────────────

class TestCustomLabelEncoder:
    def test_fit_transform_basic(self):
        enc = CustomLabelEncoder()
        result = enc.fit_transform(['a', 'b', 'a', 'c'])
        assert result == [0, 1, 0, 2]

    def test_sorted_encoding(self):
        enc = CustomLabelEncoder()
        enc.fit(['z', 'a', 'm'])
        assert enc.classes_['a'] == 0
        assert enc.classes_['m'] == 1
        assert enc.classes_['z'] == 2

    def test_unknown_value_returns_zero(self):
        enc = CustomLabelEncoder()
        enc.fit(['cat', 'dog'])
        result = enc.transform(['cat', 'unknown'])
        assert result[1] == 0

    def test_fit_transform_single_class(self):
        enc = CustomLabelEncoder()
        result = enc.fit_transform(['yes', 'yes', 'yes'])
        assert result == [0, 0, 0]

    def test_transform_preserves_order(self):
        enc = CustomLabelEncoder()
        enc.fit(['b', 'a'])
        assert enc.transform(['a', 'b', 'a']) == [0, 1, 0]


# ── CustomDecisionTree ────────────────────────────────────────────────────────

class TestCustomDecisionTree:
    def _linearly_separable(self):
        X = [[i] for i in range(100)]
        y = [0 if i < 50 else 1 for i in range(100)]
        return X, y

    def test_fit_and_predict_high_accuracy(self):
        X, y = self._linearly_separable()
        tree = CustomDecisionTree(max_depth=5)
        tree.fit(X, y)
        preds = tree.predict(X)
        accuracy = sum(p == t for p, t in zip(preds, y)) / len(y)
        assert accuracy > 0.90

    def test_predict_returns_correct_length(self):
        X, y = self._linearly_separable()
        tree = CustomDecisionTree(max_depth=3)
        tree.fit(X, y)
        preds = tree.predict(X[:10])
        assert len(preds) == 10

    def test_predict_only_binary_labels(self):
        X, y = self._linearly_separable()
        tree = CustomDecisionTree(max_depth=5)
        tree.fit(X, y)
        preds = tree.predict(X)
        assert all(p in (0, 1) for p in preds)

    def test_pure_class_leaf(self):
        X = [[1], [2], [3]]
        y = [1, 1, 1]
        tree = CustomDecisionTree(max_depth=5)
        tree.fit(X, y)
        preds = tree.predict(X)
        assert all(p == 1 for p in preds)

    def test_feature_importances_populated_after_fit(self):
        X, y = self._linearly_separable()
        tree = CustomDecisionTree(max_depth=5)
        tree.fit(X, y)
        assert tree.feature_importances is not None
        assert len(tree.feature_importances) == 1  # 1 feature in test data

    def test_two_feature_data(self):
        X = [[0, 0], [0, 1], [1, 0], [1, 1]] * 10
        y = [0, 1, 1, 0] * 10
        tree = CustomDecisionTree(max_depth=5)
        tree.fit(X, y)
        preds = tree.predict(X)
        assert len(preds) == 40


# ── CustomBaggingClassifier ───────────────────────────────────────────────────

class TestCustomBaggingClassifier:
    def _make_data(self):
        X = [[i, i * 2] for i in range(60)]
        y = [0 if i < 30 else 1 for i in range(60)]
        return X, y

    def test_fit_stores_correct_tree_count(self):
        X, y = self._make_data()
        clf = CustomBaggingClassifier(n_estimators=5)
        clf.fit(X, y)
        assert len(clf.trees) == 5

    def test_predict_returns_binary(self):
        X, y = self._make_data()
        clf = CustomBaggingClassifier(n_estimators=3)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert all(p in (0, 1) for p in preds)

    def test_predict_returns_correct_length(self):
        X, y = self._make_data()
        clf = CustomBaggingClassifier(n_estimators=3)
        clf.fit(X, y)
        preds = clf.predict(X[:10])
        assert len(preds) == 10

    def test_predict_proba_sums_to_one(self):
        X, y = self._make_data()
        clf = CustomBaggingClassifier(n_estimators=3)
        clf.fit(X, y)
        probas = clf.predict_proba(X)
        for p in probas:
            assert len(p) == 2
            assert abs(p[0] + p[1] - 1.0) < 1e-9

    def test_feature_importances_normalized(self):
        X, y = self._make_data()
        clf = CustomBaggingClassifier(n_estimators=5)
        clf.fit(X, y)
        importances = clf.feature_importances_
        assert len(importances) == 2
        total = sum(importances)
        assert total == 0.0 or abs(total - 1.0) < 1e-6

    def test_single_estimator(self):
        X, y = self._make_data()
        clf = CustomBaggingClassifier(n_estimators=1)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert len(preds) == len(X)


# ── calculate_metrics ─────────────────────────────────────────────────────────

class TestCalculateMetrics:
    def test_perfect_predictions(self):
        y_true = [1, 0, 1, 0, 1]
        y_pred = [1, 0, 1, 0, 1]
        m = calculate_metrics(y_true, y_pred)
        assert m['accuracy'] == 1.0
        assert m['precision'] == 1.0
        assert m['recall'] == 1.0
        assert m['f1_score'] == 1.0

    def test_all_wrong(self):
        y_true = [1, 1, 0, 0]
        y_pred = [0, 0, 1, 1]
        m = calculate_metrics(y_true, y_pred)
        assert m['accuracy'] == 0.0

    def test_all_predicted_negative(self):
        y_true = [1, 1, 0, 0]
        y_pred = [0, 0, 0, 0]
        m = calculate_metrics(y_true, y_pred)
        assert m['recall'] == 0.0
        assert m['precision'] == 0.0

    def test_confusion_matrix_shape(self):
        y_true = [1, 0, 1, 0]
        y_pred = [1, 0, 0, 1]
        m = calculate_metrics(y_true, y_pred)
        cm = m['confusion_matrix']
        assert len(cm) == 2 and len(cm[0]) == 2 and len(cm[1]) == 2

    def test_confusion_matrix_values(self):
        y_true = [1, 0, 1, 0]
        y_pred = [1, 1, 0, 0]
        m = calculate_metrics(y_true, y_pred)
        tn, fp = m['confusion_matrix'][0]
        fn, tp = m['confusion_matrix'][1]
        assert tp == 1
        assert tn == 1
        assert fp == 1
        assert fn == 1

    def test_f1_harmonic_mean(self):
        y_true = [1, 1, 0, 0, 1]
        y_pred = [1, 0, 0, 0, 1]
        m = calculate_metrics(y_true, y_pred)
        expected_p = 1.0
        expected_r = 2 / 3
        expected_f1 = 2 * expected_p * expected_r / (expected_p + expected_r)
        assert abs(m['f1_score'] - expected_f1) < 1e-6

    def test_metrics_range(self):
        y_true = [1, 0, 1, 0, 0, 1]
        y_pred = [1, 0, 0, 1, 0, 1]
        m = calculate_metrics(y_true, y_pred)
        for key in ('accuracy', 'precision', 'recall', 'f1_score'):
            assert 0.0 <= m[key] <= 1.0
