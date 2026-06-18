import numpy as np
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from scipy.optimize import minimize
from sklearn.metrics import f1_score
from src.config import SEED, THRESHOLD

def get_predictions(model, generator):
    preds = []
    for batch_X, _ in generator:
        p = model.predict(batch_X, verbose=0)
        preds.extend(p.ravel())
    return np.array(preds[:len(generator.labels)])

def get_feature_vectors(model, generator):
    feature_model = tf.keras.Model(
        inputs=model.input,
        outputs=model.layers[-3].output # Targets the Dense(512) layer
    )
    features = []
    for batch_X, _ in generator:
        f = feature_model.predict(batch_X, verbose=0)
        features.extend(f)
    return np.array(features[:len(generator.labels)])

def apply_smote_features(combined_features, y_train):
    smote = SMOTE(random_state=SEED, k_neighbors=5)
    X_smote, y_smote = smote.fit_resample(combined_features, y_train.astype(int))
    return X_smote, y_smote

def neg_f1_objective(weights, pred_matrix, true_labels):
    w = np.abs(weights)
    w = w / (w.sum() + 1e-8)
    ensemble_prob = pred_matrix @ w
    ensemble_pred = (ensemble_prob >= THRESHOLD).astype(int)
    return -f1_score(true_labels, ensemble_pred, zero_division=0)

def optimize_ensemble_weights(val_pred_matrix, y_val, n_models):
    w0 = np.ones(n_models) / n_models
    result = minimize(
        neg_f1_objective, x0=w0, args=(val_pred_matrix, y_val),
        method='Nelder-Mead',
        options={'maxiter': 10000, 'xatol': 1e-6, 'fatol': 1e-6, 'disp': True}
    )
    raw_weights = np.abs(result.x)
    return raw_weights / raw_weights.sum()