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
        
        # More detailed explanation based on probability ranges
        if p < 0.20:
            details = (
                f"The machine learning model has analyzed the cellular features and calculated a malignant probability of {p:.1%}. "
                "This is a very low risk score, suggesting the characteristics are strongly consistent with benign (non-cancerous) tissue. "
                "The cell features analyzed—such as radius, texture, perimeter, area, smoothness, compactness, and concavity—show patterns "
                "typical of normal or benign breast cells. However, this is a computational assessment based on numerical patterns, "
                "not a clinical diagnosis. AI models can assist healthcare professionals but cannot replace proper medical examination."
            )
        elif p < 0.40:
            details = (
                f"The model has assigned a malignant probability of {p:.1%}, which falls in the low-to-moderate range. "
                "The cellular characteristics show predominantly benign features, though some measurements may be slightly elevated "
                "or fall in borderline ranges. This doesn't necessarily indicate cancer—many benign conditions (like fibroadenomas "
                "or cysts) can produce similar patterns. The algorithm considers multiple cellular features including size, shape, "
                "texture irregularities, and boundary characteristics. While the overall assessment leans toward benign, "
                "clinical correlation with imaging and physical examination is essential for accurate diagnosis."
            )
        else:  # 0.40 to just below threshold
            details = (
                f"The calculated malignant probability is {p:.1%}, which is approaching the decision threshold of {thr:.1%}. "
                "This indicates that while the classification suggests benign, there are some cellular features that show "
                "characteristics occasionally seen in malignant cases. The model may have detected slight irregularities in "
                "nuclear shape, texture variations, or other morphological features. This is a borderline case where the "
                "distinction between benign and malignant patterns is less clear-cut. Such cases often benefit from additional "
                "diagnostic procedures to rule out any possibility of malignancy with greater certainty."
            )
        
        next_steps = (
            "🔍 **Recommended Actions:**\n"
            "1. **Clinical Correlation:** Discuss these results with your healthcare provider along with your medical history, symptoms, and physical examination findings.\n"
            "2. **Imaging Studies:** Your doctor may recommend mammography, ultrasound, or MRI to visually assess the tissue.\n"
            "3. **Regular Monitoring:** Even with low-risk results, continue regular breast self-exams and scheduled screenings.\n"
            "4. **Follow-up Timeline:** If you have any symptoms (new lumps, changes in breast tissue, discharge, or pain), "
            "schedule an appointment promptly. Otherwise, follow standard screening guidelines for your age and risk profile.\n"
            "5. **Lifestyle Factors:** Maintain healthy habits including regular exercise, balanced diet, and limiting alcohol intake."
        )
    else:
        summary = prefix + "the result looks higher risk."
        
        # More detailed explanation for malignant predictions
        if p >= 0.80:
            details = (
                f"The machine learning model has calculated a malignant probability of {p:.1%}, which is substantially above the decision threshold. "
                "This high-risk score indicates that the analyzed cellular features show patterns strongly associated with malignant (cancerous) tissue. "
                "The model likely detected significant irregularities in multiple features such as: larger and more irregular cell nuclei, "
                "increased texture roughness, irregular cell boundaries (concavity), asymmetric cell shapes, and potentially increased cell density. "
                "These characteristics are commonly observed in cancerous cells. **However, this is NOT a cancer diagnosis**—it is a risk "
                "assessment based on computational analysis. Definitive diagnosis requires clinical examination, medical imaging (mammography/ultrasound/MRI), "
                "and often histopathological examination of tissue samples (biopsy). False positives can occur, especially with certain benign conditions "
                "that mimic malignant features."
            )
        elif p >= 0.60:
            details = (
                f"The model reports a malignant probability of {p:.1%}, indicating moderate-to-high risk. "
                "The cellular features show several characteristics that raise concern for potential malignancy. The algorithm has identified "
                "patterns in the cell measurements that are more commonly associated with cancerous tissue than benign conditions. "
                "This could include irregularities in cell size and shape, increased nuclear-to-cytoplasm ratio, uneven texture distribution, "
                "or jagged cell boundaries. While these findings warrant serious attention and prompt medical evaluation, they are not definitive. "
                "Some aggressive benign lesions or atypical hyperplasia can also produce similar feature patterns. The key next step is clinical "
                "investigation to determine the true nature of these findings through additional diagnostic procedures."
            )
        else:  # Just above threshold to 0.60
            details = (
                f"The malignant probability is calculated at {p:.1%}, which exceeds the threshold of {thr:.1%} but falls in the borderline-to-moderate risk range. "
                "This suggests the presence of some cellular features that raise concern, though the evidence is not as strong as in high-confidence cases. "
                "The model may have detected subtle abnormalities such as slight irregularities in cell morphology, minor asymmetries, or borderline measurements "
                "in size and texture parameters. This zone of uncertainty is common in screening scenarios and highlights why AI tools should augment—not replace—"
                "clinical judgment. Many factors can influence these borderline results, including sample quality, tissue density, and the presence of benign "
                "conditions with overlapping features. This underscores the critical importance of comprehensive medical evaluation."
            )
        
        next_steps = (
            "🩺 **Recommended Next Steps:**\n"
            "1. **Immediate Clinical Review:** Schedule an appointment with your healthcare provider or a breast specialist (oncologist/surgical oncologist) as soon as possible. "
            "Do not delay—early detection and diagnosis significantly improve treatment outcomes.\n"
            "2. **Diagnostic Imaging:** Your physician will likely order comprehensive imaging studies:\n"
            "   • **Diagnostic Mammography:** Detailed X-ray images of the breast tissue\n"
            "   • **Breast Ultrasound:** Sound wave imaging to examine tissue density and fluid-filled vs solid masses\n"
            "   • **MRI (if indicated):** Magnetic resonance imaging for more detailed soft tissue examination\n"
            "3. **Tissue Biopsy:** If imaging shows suspicious findings, a biopsy (fine-needle aspiration, core needle biopsy, or surgical biopsy) "
            "will be performed to obtain tissue samples for microscopic examination by a pathologist. This is the gold standard for diagnosis.\n"
            "4. **Multidisciplinary Consultation:** Depending on findings, you may be referred to a team including radiologists, pathologists, "
            "oncologists, and breast surgeons for comprehensive evaluation.\n"
            "5. **Genetic Counseling (if appropriate):** If malignancy is confirmed or if you have a strong family history, genetic testing "
            "for BRCA1/BRCA2 and other cancer-related genes may be recommended.\n"
            "6. **Support Resources:** Consider reaching out to cancer support groups or patient advocacy organizations for emotional support and information.\n\n"
            "⚠️ **Important Reminder:** Please arrange a clinical review. In real clinical care, this type of result is usually followed by diagnostic imaging "
            "(mammography, ultrasound, or MRI), and often a tissue biopsy for definitive histopathological diagnosis. Early action is key."
        )

    if risk_tier == "indeterminate":
        details = (
            f"⚖️ **Borderline/Indeterminate Result:** This prediction falls very close to the decision threshold ({thr:.1%}), "
            f"with a calculated probability of {p:.1%}. This represents a diagnostically challenging case where the cellular features "
            "show a mixture of benign and potentially concerning characteristics, making clear classification difficult. "
            "In such borderline situations, the AI model's confidence is lower, and clinical context becomes even more critical. "
            "Factors such as patient age, family history, previous breast conditions, hormonal status, and the specific location and "
            "presentation of the lesion all play crucial roles in interpretation. " + details + "\n\n"
            "**Why borderline results occur:** Machine learning models are trained on historical data and learn to recognize patterns. "
            "However, biological systems are complex, and some cases naturally fall in gray zones where features overlap between benign "
            "and malignant categories. This doesn't indicate model failure—rather, it reflects the inherent complexity of breast pathology."
        )
        next_steps = (
            "🔄 **Enhanced Follow-up for Borderline Cases:**\n"
            "Because this result is indeterminate and sits close to the decision boundary, the following steps are particularly important:\n"
            "1. **Comprehensive Clinical Assessment:** A thorough evaluation by a breast specialist who can integrate this computational result "
            "with physical examination, patient history, and risk factors.\n"
            "2. **Advanced Imaging:** Consider multiple imaging modalities for cross-verification (mammography + ultrasound + possibly MRI).\n"
            "3. **Short-interval Follow-up:** Your physician may recommend a follow-up examination or imaging in 3-6 months to monitor for any changes.\n"
            "4. **Consider Additional Biomarkers:** In some cases, additional laboratory tests or molecular markers may help clarify the diagnosis.\n"
            "5. **Seek Second Opinion:** For borderline cases, consulting with multiple specialists can provide additional perspective.\n\n" + next_steps
        )

    agreement_note: Optional[str] = None
    if other_model_malignant_probability is not None:
        p2 = float(np.clip(other_model_malignant_probability, 0.0, 1.0))
        # Flag a disagreement if labels differ or probabilities are far apart.
        label2 = "malignant" if p2 >= thr else "benign"
        if label2 != label or abs(p - p2) >= 0.25:
            agreement_note = (
                f"⚠️ **Model Disagreement Detected:** The two models show significant disagreement. "
                f"The primary model calculated {p:.1%} malignant probability (classified as **{label}**), "
                f"while the {other_model_name} calculated {p2:.1%} (classified as **{label2}**). "
                f"This disagreement suggests that the case may have ambiguous or conflicting features that different algorithms interpret differently. "
                "\n\n**What this means:** When machine learning models disagree, it often indicates a complex case that doesn't fit neatly into typical patterns. "
                "This could be due to: (1) unusual combinations of cellular features, (2) image quality or measurement variability, "
                "(3) the presence of rare or atypical pathology, or (4) features that genuinely fall in a gray zone between benign and malignant. "
                "\n\n**Clinical Action Required:** Model disagreement is a strong signal that this case requires careful clinical review. "
                "Do NOT rely solely on the computational prediction. It is essential to follow up with a clinician who can integrate these results "
                "with imaging studies, physical examination findings, patient history, and additional diagnostic tests. "
                "In cases of model disagreement, clinicians often proceed with more cautious or comprehensive evaluation, "
                "potentially including second opinions from multiple specialists or additional tissue sampling to ensure accurate diagnosis."
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
