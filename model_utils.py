"""Shared ML utilities for the Bio2 Mini Project Streamlit app.

This module holds dataset loading, training, and threshold selection logic.
The Streamlit pages import from here so UI can be split cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def explain_prediction(
    malignant_probability: float,
    *,
    threshold: float = 0.5,
    patient_name: Optional[str] = None,
    other_model_malignant_probability: Optional[float] = None,
    other_model_name: str = "second model",
) -> Dict[str, object]:
    """Create a doctor-style explanation for a single prediction.

    This is intentionally rule-based. It doesn't pretend to be clinical guidance,
    but it helps translate a probability + threshold into patient-friendly text.

    Args:
        malignant_probability: Model probability for malignant class in [0, 1].
        threshold: Decision cutoff used to label malignant vs benign.
        patient_name: Optional name to personalize the message.
        other_model_malignant_probability: If provided, we will mention if the
            second model strongly disagrees.
        other_model_name: Display name for the second model.

    Returns:
        Dict with keys:
          - label: 'benign' or 'malignant'
          - risk_tier: 'low' | 'moderate' | 'high' | 'indeterminate'
          - summary: one-liner
          - details: longer explanation
          - next_steps: actionable but non-prescriptive suggestions
          - malignant_probability: float
          - threshold: float
          - agreement_note: optional str
    """

    p = float(np.clip(malignant_probability, 0.0, 1.0))
    thr = float(np.clip(threshold, 0.0, 1.0))
    label = "malignant" if p >= thr else "benign"

    # Probability tiers: keep simple and easy to explain.
    # We treat the region near the threshold as "indeterminate" to encourage follow-up.
    delta = abs(p - thr)
    if delta <= 0.10:
        risk_tier = "indeterminate"
    elif p >= 0.70:
        risk_tier = "high"
    elif p >= 0.40:
        risk_tier = "moderate"
    else:
        risk_tier = "low"

    # Human-facing text
    prefix = f"{patient_name}, " if patient_name else ""

    if label == "benign":
        summary = prefix + "the result looks low risk."
        details = (
            "The model estimates a low chance that this is malignant. "
            "That’s reassuring, but it’s not a final diagnosis."
        )
        next_steps = (
            "If you have symptoms, a new lump, or ongoing worry, please see a clinician for a proper exam and imaging."
        )
    else:
        summary = prefix + "the result looks higher risk."
        details = (
            "The model estimates a higher chance that this is malignant. "
            "This does not confirm cancer, but it means you should get checked soon."
        )
        next_steps = (
            "Please arrange a clinical review. In real care this is usually followed by imaging, and sometimes a biopsy."
        )

    if risk_tier == "indeterminate":
        details = "This result is close to the cutoff, so it’s not clear-cut. " + details
        next_steps = "Because it’s borderline, follow-up and clinical context matter more. " + next_steps

    agreement_note: Optional[str] = None
    if other_model_malignant_probability is not None:
        p2 = float(np.clip(other_model_malignant_probability, 0.0, 1.0))
        # Flag a disagreement if labels differ or probabilities are far apart.
        label2 = "malignant" if p2 >= thr else "benign"
        if label2 != label or abs(p - p2) >= 0.25:
            agreement_note = (
                f"The two models do not agree ({other_model_name} differs). "
                "When models disagree, it’s safer to follow up with a clinician and confirm with proper tests."
            )

    return {
        "label": label,
        "risk_tier": risk_tier,
        "summary": summary,
        "details": details,
        "next_steps": next_steps,
        "malignant_probability": p,
        "threshold": thr,
        "agreement_note": agreement_note,
    }


def _rng(seed: int) -> np.random.RandomState:
    # Keep deterministic behavior for reproducibility.
    return np.random.RandomState(int(seed))


@dataclass(frozen=True)
class TrainConfig:
    test_size: float = 0.2
    random_state: int = 42

    # Model choice
    model_type: str = "random_forest"  # random_forest | extra_trees | log_reg | svm_rbf | grad_boost

    # Tree models
    n_estimators: int = 300
    max_depth: int = 6
    min_samples_leaf: int = 2

    # Linear/SVM models
    logreg_c: float = 1.0
    svm_c: float = 2.0
    svm_gamma: str = "scale"  # scale | auto
    calibrate_probabilities: bool = True
    cv_folds: int = 5
    threshold_strategy: str = "target_sensitivity"  # target_sensitivity | youden | fixed
    target_sensitivity: float = 0.95
    fixed_threshold: float = 0.50


FEATURES = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean compactness",
    "mean concavity",
    "mean concave points",
    "mean symmetry",
    "mean fractal dimension",
]


def _specificity(cm: np.ndarray) -> float:
    tn, fp = cm[0, 0], cm[0, 1]
    return float(tn / (tn + fp)) if (tn + fp) else 0.0


def _youden_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    thresholds = np.unique(np.clip(y_score, 0, 1))
    best_thr = 0.5
    best_j = -1e9
    for thr in thresholds:
        y_pred = (y_score >= thr).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        j = tpr - fpr
        if j > best_j:
            best_j = j
            best_thr = float(thr)
    return best_thr


def _target_sensitivity_threshold(y_true: np.ndarray, y_score: np.ndarray, target: float) -> float:
    thresholds = np.unique(np.clip(y_score, 0, 1))
    thresholds.sort()
    best = thresholds[0]
    for thr in thresholds:
        y_pred = (y_score >= thr).astype(int)
        rec = recall_score(y_true, y_pred, zero_division=0)
        if rec >= target:
            best = thr
    return float(best)


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    # In sklearn dataset: target 0=malignant, 1=benign
    df["diagnosis"] = df["target"].map({0: "M", 1: "B"})
    return df


@st.cache_resource(show_spinner=False)
def train_model(cfg: TrainConfig) -> Tuple[object, StandardScaler, Dict[str, float], float, pd.Series]:
    """Train model and return (model, scaler, metrics, threshold, feature_medians)."""
    df = load_dataset()
    X = df[FEATURES]
    y = df["target"].astype(int)  # 0 malignant, 1 benign

    # We'll use positive class = malignant for sensitivity.
    y_malig = (y == 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_malig,
        test_size=float(cfg.test_size),
        random_state=int(cfg.random_state),
        stratify=y_malig,
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    base = build_base_model(cfg)

    if cfg.calibrate_probabilities:
        cv = StratifiedKFold(n_splits=int(cfg.cv_folds), shuffle=True, random_state=int(cfg.random_state))
        model = CalibratedClassifierCV(base, method="isotonic", cv=cv)
    else:
        model = base

    model.fit(X_train_s, y_train)
    y_score = model.predict_proba(X_test_s)[:, 1]

    if cfg.threshold_strategy == "fixed":
        thr = float(cfg.fixed_threshold)
    elif cfg.threshold_strategy == "youden":
        thr = _youden_threshold(y_test.to_numpy(), y_score)
    else:
        thr = _target_sensitivity_threshold(y_test.to_numpy(), y_score, float(cfg.target_sensitivity))

    y_pred = (y_score >= thr).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_score)),
        "specificity": float(_specificity(cm)),
        "threshold": float(thr),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }

    feature_medians = X_train.median(numeric_only=True)
    return model, scaler, metrics, float(thr), feature_medians


def build_base_model(cfg: TrainConfig):
    """Create an uncalibrated estimator according to cfg.model_type."""
    mt = str(cfg.model_type)

    if mt == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(cfg.n_estimators),
            max_depth=int(cfg.max_depth),
            min_samples_leaf=int(cfg.min_samples_leaf),
            random_state=int(cfg.random_state),
            class_weight="balanced",
            n_jobs=-1,
        )

    if mt == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=int(cfg.n_estimators),
            max_depth=int(cfg.max_depth),
            min_samples_leaf=int(cfg.min_samples_leaf),
            random_state=int(cfg.random_state),
            class_weight="balanced",
            n_jobs=-1,
        )

    if mt == "grad_boost":
        # GradientBoosting doesn't support class_weight; keep it simple.
        return GradientBoostingClassifier(random_state=int(cfg.random_state))

    if mt == "log_reg":
        # Robust baseline for tabular data.
        return LogisticRegression(
            C=float(cfg.logreg_c),
            max_iter=5000,
            solver="lbfgs",
            class_weight="balanced",
        )

    if mt == "svm_rbf":
        # Enable probabilities for thresholding; SVC is slower but can perform well.
        return SVC(
            C=float(cfg.svm_c),
            gamma=str(cfg.svm_gamma),
            probability=True,
            class_weight="balanced",
            random_state=int(cfg.random_state),
        )

    raise ValueError(f"Unknown model_type: {mt}")


def _sample_cfg(base_cfg: TrainConfig, r: np.random.RandomState) -> TrainConfig:
    """Randomly sample a new TrainConfig (hyperparameter variant).

    This is a simple random search (not exhaustive). It’s designed to allow
    training *many* models (e.g., 100+) quickly and pick the best by a metric.
    """
    mt = str(r.choice(["random_forest", "extra_trees", "log_reg", "svm_rbf", "grad_boost"]))

    # Keep threshold/calibration strategy from base_cfg unless user changes it.
    d = dict(base_cfg.__dict__)
    d["model_type"] = mt

    if mt in {"random_forest", "extra_trees"}:
        d["n_estimators"] = int(r.choice([200, 300, 500, 800, 1200]))
        d["max_depth"] = int(r.choice([3, 4, 5, 6, 8, 10, 14, 18]))
        d["min_samples_leaf"] = int(r.choice([1, 2, 3, 4, 5]))

    if mt == "log_reg":
        # Log-uniform sampling for C
        d["logreg_c"] = float(10 ** r.uniform(-2, 1))  # 0.01 to 10

    if mt == "svm_rbf":
        d["svm_c"] = float(10 ** r.uniform(-1, 1.3))  # ~0.1 to ~20
        d["svm_gamma"] = str(r.choice(["scale", "auto"]))

    # For grad_boost, we keep default parameters (could be extended later).
    return TrainConfig(**d)


@st.cache_resource(show_spinner=False)
def random_model_search(
    base_cfg: TrainConfig,
    n_trials: int = 100,
    primary_metric: str = "roc_auc",
    top_k: int = 10,
) -> Tuple[TrainConfig, object, StandardScaler, Dict[str, float], float, pd.Series, pd.DataFrame]:
    """Train many models and return the best plus a leaderboard.

    Returns:
      (best_cfg, best_model, best_scaler, best_metrics, best_threshold, feature_medians, leaderboard_df)

    Notes:
    - Uses a single train/test split (determined by base_cfg.test_size/base_cfg.random_state)
    - Each trial samples a model family + hyperparameters.
    - Calibration and threshold strategy follow base_cfg.
    """
    df = load_dataset()
    X = df[FEATURES]
    y = df["target"].astype(int)
    y_malig = (y == 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_malig,
        test_size=float(base_cfg.test_size),
        random_state=int(base_cfg.random_state),
        stratify=y_malig,
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    feature_medians = X_train.median(numeric_only=True)

    r = _rng(base_cfg.random_state)
    rows: List[Dict[str, object]] = []

    best_score = -1e18
    best_cfg: Optional[TrainConfig] = None
    best_model = None
    best_metrics: Optional[Dict[str, float]] = None
    best_thr: float = 0.5

    for i in range(int(n_trials)):
        cfg = _sample_cfg(base_cfg, r)
        base = build_base_model(cfg)

        if cfg.calibrate_probabilities:
            cv = StratifiedKFold(n_splits=int(cfg.cv_folds), shuffle=True, random_state=int(cfg.random_state))
            model = CalibratedClassifierCV(base, method="isotonic", cv=cv)
        else:
            model = base

        model.fit(X_train_s, y_train)
        y_score = model.predict_proba(X_test_s)[:, 1]

        if cfg.threshold_strategy == "fixed":
            thr = float(cfg.fixed_threshold)
        elif cfg.threshold_strategy == "youden":
            thr = _youden_threshold(y_test.to_numpy(), y_score)
        else:
            thr = _target_sensitivity_threshold(y_test.to_numpy(), y_score, float(cfg.target_sensitivity))

        y_pred = (y_score >= thr).astype(int)
        cm = confusion_matrix(y_test, y_pred)

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_score)),
            "specificity": float(_specificity(cm)),
            "threshold": float(thr),
        }

        score = float(metrics.get(primary_metric, 0.0))
        rows.append(
            {
                "trial": i + 1,
                "model_type": cfg.model_type,
                "score": score,
                "roc_auc": metrics["roc_auc"],
                "accuracy": metrics["accuracy"],
                "f1": metrics["f1"],
                "recall": metrics["recall"],
                "specificity": metrics["specificity"],
                "threshold": metrics["threshold"],
                # useful params for trees / linear
                "n_estimators": getattr(cfg, "n_estimators", None),
                "max_depth": getattr(cfg, "max_depth", None),
                "min_samples_leaf": getattr(cfg, "min_samples_leaf", None),
                "logreg_c": getattr(cfg, "logreg_c", None),
                "svm_c": getattr(cfg, "svm_c", None),
                "svm_gamma": getattr(cfg, "svm_gamma", None),
            }
        )

        if score > best_score:
            best_score = score
            best_cfg = cfg
            best_model = model
            best_metrics = metrics
            best_thr = float(thr)

    leaderboard = pd.DataFrame(rows).sort_values(by="score", ascending=False).head(int(top_k))
    assert best_cfg is not None and best_metrics is not None and best_model is not None
    return best_cfg, best_model, scaler, best_metrics, best_thr, feature_medians, leaderboard


@st.cache_resource(show_spinner=False)
def train_and_compare_models(
    cfg: TrainConfig,
    model_types: Tuple[str, ...],
    primary_metric: str = "roc_auc",
) -> Tuple[str, Dict[str, Dict[str, float]]]:
    """Train multiple models (same split) and return (best_model_type, metrics_by_model).

    Notes:
    - Uses the same train/test split determined by cfg.test_size and cfg.random_state.
    - Each model uses its own configuration via cfg.model_type override.
    """
    df = load_dataset()
    X = df[FEATURES]
    y = df["target"].astype(int)
    y_malig = (y == 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_malig,
        test_size=float(cfg.test_size),
        random_state=int(cfg.random_state),
        stratify=y_malig,
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    metrics_by_model: Dict[str, Dict[str, float]] = {}
    best_mt: Optional[str] = None
    best_score = -1e18

    for mt in model_types:
        local_cfg = TrainConfig(**{**cfg.__dict__, "model_type": mt})
        base = build_base_model(local_cfg)

        if local_cfg.calibrate_probabilities:
            cv = StratifiedKFold(n_splits=int(local_cfg.cv_folds), shuffle=True, random_state=int(local_cfg.random_state))
            model = CalibratedClassifierCV(base, method="isotonic", cv=cv)
        else:
            model = base

        model.fit(X_train_s, y_train)
        y_score = model.predict_proba(X_test_s)[:, 1]

        # Use same threshold strategy as train_model
        if local_cfg.threshold_strategy == "fixed":
            thr = float(local_cfg.fixed_threshold)
        elif local_cfg.threshold_strategy == "youden":
            thr = _youden_threshold(y_test.to_numpy(), y_score)
        else:
            thr = _target_sensitivity_threshold(y_test.to_numpy(), y_score, float(local_cfg.target_sensitivity))

        y_pred = (y_score >= thr).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        m = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_score)),
            "specificity": float(_specificity(cm)),
            "threshold": float(thr),
        }
        metrics_by_model[mt] = m

        score = float(m.get(primary_metric, 0.0))
        if score > best_score:
            best_score = score
            best_mt = mt

    return best_mt or model_types[0], metrics_by_model
