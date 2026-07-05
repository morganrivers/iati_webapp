"""
Minimal treeinterpreter replacement compatible with modern scikit-learn.

treeinterpreter breaks because tree_.value now has shape (n_nodes, 1, 1)
instead of (n_nodes,). This reimplements the same path-based algorithm
using only numpy + sklearn's built-in decision_path.

API: prediction, bias, contributions = predict(rf_model, X)
  - prediction  : (n_samples,)  - model prediction per sample
  - bias        : (n_samples,)  - root-node value (training mean) per sample
  - contributions: (n_samples, n_features) - signed per-feature contribution
"""
import numpy as np


def _predict_tree(tree, X):
    values = tree.tree_.value[:, 0, 0]          # (n_nodes,) for single-output regressor
    feature = tree.tree_.feature                 # split feature at each node (-2 = leaf)
    node_indicator = tree.decision_path(X)       # sparse (n_samples, n_nodes)

    n_samples = X.shape[0]
    n_features = tree.n_features_in_
    preds   = np.empty(n_samples)
    biases  = np.empty(n_samples)
    contribs = np.zeros((n_samples, n_features))

    for i in range(n_samples):
        path = node_indicator[i].indices         # node ids root → leaf
        biases[i]  = values[path[0]]
        preds[i]   = values[path[-1]]
        for j in range(len(path) - 1):
            contribs[i, feature[path[j]]] += values[path[j + 1]] - values[path[j]]

    return preds, biases, contribs


def predict(model, X):
    """Drop-in replacement for treeinterpreter.predict for RandomForestRegressor."""
    n_trees = len(model.estimators_)
    n_samples, n_features = X.shape[0], X.shape[1]
    sum_preds    = np.zeros(n_samples)
    sum_biases   = np.zeros(n_samples)
    sum_contribs = np.zeros((n_samples, n_features))

    for tree in model.estimators_:
        p, b, c = _predict_tree(tree, X)
        sum_preds    += p
        sum_biases   += b
        sum_contribs += c

    return sum_preds / n_trees, sum_biases / n_trees, sum_contribs / n_trees
