"""Single-file Streamlit app for breast cancer classification.

This file contains both:
1) Streamlit UI
2) ML pipeline (leakage-safe pipelines, CV tuning, calibration, thresholding)

Run:
    streamlit run app.py
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def _specificity_from_cm(cm: np.ndarray) -> float:
    tn, fp = cm[0, 0], cm[0, 1]
    denom = tn + fp
    return float(tn / denom) if denom else 0.0


@dataclass
class TrainConfig:
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5

    # Clinical-ish knobs
    positive_label: int = 1  # 1 = malignant
    target_sensitivity: float = 0.95
    calibrate_probabilities: bool = True
    threshold_strategy: str = "target_sensitivity"  # or "youden" or "fixed"
    fixed_threshold: float = 0.5


class BreastCancerClassifier:
    """Train and evaluate Logistic Regression vs tuned SVM."""

    def __init__(self, data_path: str | Path = "breast_cancer_data.csv"):
        self.data_path = Path(data_path)

        self.df: Optional[pd.DataFrame] = None
        self.X: Optional[pd.DataFrame] = None
        self.y: Optional[pd.Series] = None

        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None

        self.feature_names: Optional[List[str]] = None

        self.lr_model: Optional[Any] = None
        self.svm_model: Optional[Any] = None
        self.svm_grid: Optional[GridSearchCV] = None

        self.lr_threshold_: float = 0.5
        self.svm_threshold_: float = 0.5

        self.results: Dict[str, Any] = {
            "run_info": {},
            "dataset_info": {},
            "training_config": {},
            "interpretability": {},
            "models": {},
        }

    def load_data(self) -> pd.DataFrame:
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.data_path.resolve()} (expected CSV in project folder)"
            )

        df = pd.read_csv(self.data_path)
        self.df = df

        if "diagnosis" not in df.columns:
            raise ValueError("Expected a 'diagnosis' column in the dataset.")

        feature_cols = [c for c in df.columns if c not in {"id", "diagnosis"}]
        if not feature_cols:
            raise ValueError("No feature columns found (expected columns besides id/diagnosis).")

        self.X = df[feature_cols]
        self.feature_names = feature_cols

        # Map M/B to 1/0 for positive class = malignant
        self.y = df["diagnosis"].map({"B": 0, "M": 1})
        if self.y.isna().any():
            raise ValueError("Unexpected label values in 'diagnosis' (expected 'B'/'M').")

        self.results["run_info"] = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "data_path": str(self.data_path),
        }
        self.results["dataset_info"] = {
            "total_samples": int(df.shape[0]),
            "num_features": int(self.X.shape[1]),
            "class_distribution": {
                "benign": int((self.y == 0).sum()),
                "malignant": int((self.y == 1).sum()),
            },
        }
        return df

    def preprocess_data(self, test_size: float = 0.2, random_state: int = 42) -> None:
        if self.X is None or self.y is None:
            raise RuntimeError("Call load_data() before preprocess_data().")

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=test_size,
            random_state=random_state,
            stratify=self.y,
        )

        self.results["dataset_info"].update(
            {
                "train_samples": int(self.X_train.shape[0]),
                "test_samples": int(self.X_test.shape[0]),
                "test_size": float(test_size),
                "random_state": int(random_state),
            }
        )

    def _build_lr_pipeline(self) -> Pipeline:
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=5000, solver="lbfgs")),
            ]
        )

    def _build_svm_pipeline(self) -> Pipeline:
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", SVC(kernel="rbf", probability=True)),
            ]
        )

    def _maybe_calibrate(self, estimator: Any, config: TrainConfig) -> Any:
        if not config.calibrate_probabilities:
            return estimator
        cv = StratifiedKFold(
            n_splits=int(config.cv_folds),
            shuffle=True,
            random_state=int(config.random_state),
        )
        return CalibratedClassifierCV(estimator, method="isotonic", cv=cv)

    def _choose_threshold(self, y_true: np.ndarray, y_score: np.ndarray, config: TrainConfig) -> float:
        if config.threshold_strategy == "fixed":
            return float(config.fixed_threshold)

        y_score = np.asarray(y_score).ravel()

        if config.threshold_strategy == "youden":
            fpr, tpr, thr = roc_curve(y_true, y_score)
            j = tpr - fpr
            idx = int(np.nanargmax(j))
            return float(thr[idx])

        # default: target sensitivity on training set
        _, recall, thr = precision_recall_curve(y_true, y_score)
        if thr.size == 0:
            return 0.5
        recall_at_thr = recall[:-1]
        ok = np.where(recall_at_thr >= float(config.target_sensitivity))[0]
        if ok.size == 0:
            idx = int(np.nanargmax(recall_at_thr))
            return float(thr[idx])
        return float(thr[ok[-1]])

    @staticmethod
    def predict_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        if hasattr(model, "decision_function"):
            scores = model.decision_function(X)
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
            return scores
        raise TypeError("Model does not support probability or decision scores.")

    def train_baseline_model(self, config: TrainConfig = TrainConfig()) -> Any:
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Call preprocess_data() before training.")

        base = self._build_lr_pipeline()
        self.lr_model = self._maybe_calibrate(base, config)
        self.lr_model.fit(self.X_train, self.y_train)

        y_score_train = self.predict_proba(self.lr_model, self.X_train)
        self.lr_threshold_ = self._choose_threshold(self.y_train.to_numpy(), y_score_train, config)
        return self.lr_model

    def train_svm_model(self, config: TrainConfig = TrainConfig()) -> Any:
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Call preprocess_data() before training.")

        param_grid = {
            "model__C": [0.5, 1, 5, 10, 50],
            "model__gamma": ["scale", 0.01, 0.1, 1],
        }

        base = self._build_svm_pipeline()
        cv = StratifiedKFold(
            n_splits=int(config.cv_folds),
            shuffle=True,
            random_state=int(config.random_state),
        )
        self.svm_grid = GridSearchCV(
            base,
            param_grid,
            cv=cv,
            n_jobs=-1,
            scoring="roc_auc",
            refit=True,
        )
        self.svm_grid.fit(self.X_train, self.y_train)

        best = self.svm_grid.best_estimator_
        self.svm_model = self._maybe_calibrate(best, config)
        self.svm_model.fit(self.X_train, self.y_train)

        y_score_train = self.predict_proba(self.svm_model, self.X_train)
        self.svm_threshold_ = self._choose_threshold(self.y_train.to_numpy(), y_score_train, config)
        return self.svm_model

    def _evaluate_one(self, model: Any, threshold: float) -> Dict[str, Any]:
        if self.X_test is None or self.y_test is None:
            raise RuntimeError("Call preprocess_data() before evaluation.")

        y_score = self.predict_proba(model, self.X_test)
        y_pred = (y_score >= float(threshold)).astype(int)
        cm = confusion_matrix(self.y_test, y_pred)

        out: Dict[str, Any] = {
            "accuracy": float(accuracy_score(self.y_test, y_pred)),
            "precision": float(precision_score(self.y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(self.y_test, y_pred, zero_division=0)),
            "f1_score": float(f1_score(self.y_test, y_pred, zero_division=0)),
            "avg_precision": float(average_precision_score(self.y_test, y_score)),
            "roc_auc": float(roc_auc_score(self.y_test, y_score)),
            "brier_score": float(brier_score_loss(self.y_test, y_score)),
            "specificity": float(_specificity_from_cm(cm)),
            "threshold": float(threshold),
            "confusion_matrix": {
                "true_negatives": int(cm[0, 0]),
                "false_positives": int(cm[0, 1]),
                "false_negatives": int(cm[1, 0]),
                "true_positives": int(cm[1, 1]),
            },
            "classification_report": classification_report(
                self.y_test,
                y_pred,
                target_names=["benign", "malignant"],
                zero_division=0,
                output_dict=True,
            ),
        }
        return out

    def top_features_lr(self, top_k: int = 10) -> Dict[str, float]:
        if self.lr_model is None or self.feature_names is None:
            return {}

        model = self.lr_model
        if hasattr(model, "base_estimator"):
            model = model.base_estimator
        if not hasattr(model, "named_steps"):
            return {}

        lr = model.named_steps.get("model")
        if lr is None or not hasattr(lr, "coef_"):
            return {}

        coefs = lr.coef_.ravel()
        pairs = list(zip(self.feature_names, coefs))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return {k: float(v) for k, v in pairs[: int(top_k)]}

    def predict_patient(self, features: Dict[str, float], model_name: str = "SVM (Optimized)") -> Dict[str, Any]:
        if self.feature_names is None:
            raise RuntimeError("Call load_data() first to get feature schema.")

        missing = [f for f in self.feature_names if f not in features]
        if missing:
            raise ValueError(f"Missing feature(s): {missing[:5]}" + ("..." if len(missing) > 5 else ""))

        X_one = pd.DataFrame([{f: float(features[f]) for f in self.feature_names}])

        if model_name == "Logistic Regression":
            if self.lr_model is None:
                raise RuntimeError("Train the Logistic Regression model first.")
            p = float(self.predict_proba(self.lr_model, X_one)[0])
            thr = float(self.lr_threshold_)
        else:
            if self.svm_model is None:
                raise RuntimeError("Train the SVM model first.")
            p = float(self.predict_proba(self.svm_model, X_one)[0])
            thr = float(self.svm_threshold_)

        pred = int(p >= thr)
        return {
            "model": model_name,
            "prob_malignant": p,
            "threshold": thr,
            "predicted_label": pred,
            "predicted_class": "malignant" if pred == 1 else "benign",
        }

    def evaluate_models(self) -> Dict[str, Any]:
        if self.lr_model is None or self.svm_model is None:
            raise RuntimeError("Train both models before evaluate_models().")

        self.results["models"] = {
            "Logistic Regression": self._evaluate_one(self.lr_model, threshold=self.lr_threshold_),
            "SVM (Optimized)": self._evaluate_one(self.svm_model, threshold=self.svm_threshold_),
        }
        if self.svm_grid is not None:
            self.results["models"]["SVM (Optimized)"]["best_params"] = self.svm_grid.best_params_
        return self.results

    def save_results(self, out_path: str | Path = "results.json") -> Path:
        out_path = Path(out_path)
        out_path.write_text(json.dumps(self.results, indent=4), encoding="utf-8")
        return out_path

    def run(self, config: TrainConfig = TrainConfig()) -> Dict[str, Any]:
        self.load_data()
        self.preprocess_data(test_size=config.test_size, random_state=config.random_state)
        self.train_baseline_model(config=config)
        self.train_svm_model(config=config)
        self.evaluate_models()
        self.results["training_config"] = {
            "test_size": float(config.test_size),
            "random_state": int(config.random_state),
            "cv_folds": int(config.cv_folds),
            "calibrate_probabilities": bool(config.calibrate_probabilities),
            "threshold_strategy": str(config.threshold_strategy),
            "target_sensitivity": float(config.target_sensitivity),
            "fixed_threshold": float(config.fixed_threshold),
        }
        self.results["interpretability"] = {
            "lr_top_positive_coefficients": self.top_features_lr(top_k=10)
        }
        return self.results

# Page configuration
st.set_page_config(
    page_title="Breast Cancer Classifier",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
    <style>
    .main {
        padding-top: 0rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("🧬 Breast Cancer Classification System")
st.markdown("### Wisconsin Breast Cancer Dataset Analysis")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    st.write("Control the analysis parameters below")
    
    test_size = st.slider(
        "Test Set Size (%)",
        min_value=10,
        max_value=40,
        value=20,
        step=5,
        help="Percentage of data to use for testing"
    ) / 100
    
    random_state = st.number_input(
        "Random State",
        min_value=0,
        max_value=1000,
        value=42,
        help="For reproducibility"
    )

    st.subheader("Clinical-ish settings")

    cv_folds = st.slider(
        "Cross-validation folds",
        min_value=3,
        max_value=10,
        value=5,
        step=1,
        help="Used for SVM tuning and optional probability calibration",
    )

    calibrate_probabilities = st.toggle(
        "Calibrate probabilities (isotonic)",
        value=True,
        help="More realistic risk estimates; a bit slower",
    )

    threshold_strategy = st.selectbox(
        "Decision threshold",
        options=["target_sensitivity", "youden", "fixed"],
        index=0,
        help="Real screening often prioritizes high sensitivity (recall for malignant)",
    )

    target_sensitivity = st.slider(
        "Target sensitivity (recall for malignant)",
        min_value=0.80,
        max_value=0.99,
        value=0.95,
        step=0.01,
        help="Only used when threshold strategy is target_sensitivity",
    )

    fixed_threshold = st.slider(
        "Fixed threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.05,
        help="Only used when threshold strategy is fixed",
    )
    
    run_analysis = st.button("🚀 Run Analysis", key="run_button", use_container_width=True)

# Main content
if run_analysis:
    with st.spinner("🔄 Running analysis... This may take a few minutes..."):
        try:
            # Initialize classifier
            classifier = BreastCancerClassifier(data_path='breast_cancer_data.csv')

            config = TrainConfig(
                test_size=float(test_size),
                random_state=int(random_state),
                cv_folds=int(cv_folds),
                calibrate_probabilities=bool(calibrate_probabilities),
                threshold_strategy=str(threshold_strategy),
                target_sensitivity=float(target_sensitivity),
                fixed_threshold=float(fixed_threshold),
            )
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Load data
            status_text.text("📊 Loading dataset...")
            progress_bar.progress(10)
            classifier.load_data()
            
            # Preprocess
            status_text.text("🔧 Preprocessing data...")
            progress_bar.progress(30)
            classifier.preprocess_data(test_size=test_size, random_state=int(random_state))
            
            # Train baseline
            status_text.text("🤖 Training Logistic Regression...")
            progress_bar.progress(50)
            classifier.train_baseline_model(config=config)
            
            # Train SVM
            status_text.text("⚡ Training SVM with GridSearchCV...")
            progress_bar.progress(75)
            classifier.train_svm_model(config=config)
            
            # Evaluate
            status_text.text("📈 Evaluating models...")
            progress_bar.progress(90)
            classifier.evaluate_models()
            
            # Save results
            status_text.text("💾 Saving results...")
            progress_bar.progress(95)
            classifier.save_results()
            
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            
            st.success("✅ Analysis completed successfully!")
            st.markdown("---")
            
            # Display Results
            st.header("📊 Results Summary")
            
            # Dataset Info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Samples", classifier.results['dataset_info']['total_samples'])
            with col2:
                st.metric("Training Samples", classifier.results['dataset_info']['train_samples'])
            with col3:
                st.metric("Test Samples", classifier.results['dataset_info']['test_samples'])
            with col4:
                st.metric("Features", classifier.results['dataset_info']['num_features'])
            
            st.markdown("---")
            
            # Model Comparison
            st.header("🎯 Model Performance Comparison")
            
            col1, col2 = st.columns(2)
            
            # Logistic Regression
            with col1:
                st.subheader("Logistic Regression")
                lr_metrics = classifier.results['models']['Logistic Regression']
                
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.metric("Accuracy", f"{lr_metrics['accuracy']:.4f}")
                    st.metric("Precision", f"{lr_metrics['precision']:.4f}")
                with metric_col2:
                    st.metric("Recall", f"{lr_metrics['recall']:.4f}")
                    st.metric("F1-Score", f"{lr_metrics['f1_score']:.4f}")
                
                st.metric("ROC-AUC", f"{lr_metrics['roc_auc']:.4f}")
                st.metric("Avg Precision", f"{lr_metrics['avg_precision']:.4f}")
                st.metric("Specificity", f"{lr_metrics['specificity']:.4f}")
                st.metric("Brier score", f"{lr_metrics['brier_score']:.4f}")
                st.caption(f"Threshold used: {lr_metrics.get('threshold', 0.5):.3f}")
            
            # SVM
            with col2:
                st.subheader("SVM (Optimized)")
                svm_metrics = classifier.results['models']['SVM (Optimized)']
                
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.metric("Accuracy", f"{svm_metrics['accuracy']:.4f}")
                    st.metric("Precision", f"{svm_metrics['precision']:.4f}")
                with metric_col2:
                    st.metric("Recall", f"{svm_metrics['recall']:.4f}")
                    st.metric("F1-Score", f"{svm_metrics['f1_score']:.4f}")
                
                st.metric("ROC-AUC", f"{svm_metrics['roc_auc']:.4f}")
                st.metric("Avg Precision", f"{svm_metrics['avg_precision']:.4f}")
                st.metric("Specificity", f"{svm_metrics['specificity']:.4f}")
                st.metric("Brier score", f"{svm_metrics['brier_score']:.4f}")
                st.caption(f"Threshold used: {svm_metrics.get('threshold', 0.5):.3f}")

                if 'best_params' in svm_metrics:
                    st.caption(f"Best params: {svm_metrics['best_params']}")
            
            st.markdown("---")
            
            # Visualizations
            st.header("📈 Visualizations")
            
            # Create visualizations
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Breast Cancer Classification - Model Comparison', fontsize=16, fontweight='bold')
            
            models = {
                'Logistic Regression': classifier.lr_model,
                'SVM (Optimized)': classifier.svm_model
            }
            
            # 1. Performance Metrics Comparison
            ax1 = axes[0, 0]
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            lr_values = [
                classifier.results['models']['Logistic Regression']['accuracy'],
                classifier.results['models']['Logistic Regression']['precision'],
                classifier.results['models']['Logistic Regression']['recall'],
                classifier.results['models']['Logistic Regression']['f1_score']
            ]
            svm_values = [
                classifier.results['models']['SVM (Optimized)']['accuracy'],
                classifier.results['models']['SVM (Optimized)']['precision'],
                classifier.results['models']['SVM (Optimized)']['recall'],
                classifier.results['models']['SVM (Optimized)']['f1_score']
            ]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            ax1.bar(x - width/2, lr_values, width, label='Logistic Regression', color='skyblue')
            ax1.bar(x + width/2, svm_values, width, label='SVM (Optimized)', color='lightcoral')
            ax1.set_xlabel('Metrics')
            ax1.set_ylabel('Score')
            ax1.set_title('Performance Metrics Comparison')
            ax1.set_xticks(x)
            ax1.set_xticklabels(metrics, rotation=45, ha='right')
            ax1.legend()
            ax1.grid(axis='y', alpha=0.3)
            ax1.set_ylim([0.85, 1.0])
            
            # 2. Confusion Matrix - Logistic Regression
            ax2 = axes[0, 1]
            cm_lr = [
                [classifier.results['models']['Logistic Regression']['confusion_matrix']['true_negatives'],
                 classifier.results['models']['Logistic Regression']['confusion_matrix']['false_positives']],
                [classifier.results['models']['Logistic Regression']['confusion_matrix']['false_negatives'],
                 classifier.results['models']['Logistic Regression']['confusion_matrix']['true_positives']]
            ]
            sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=ax2, cbar=False)
            ax2.set_title('Confusion Matrix - Logistic Regression')
            ax2.set_xlabel('Predicted')
            ax2.set_ylabel('Actual')
            ax2.set_xticklabels(['Benign', 'Malignant'])
            ax2.set_yticklabels(['Benign', 'Malignant'], rotation=0)
            
            # 3. Confusion Matrix - SVM
            ax3 = axes[1, 0]
            cm_svm = [
                [classifier.results['models']['SVM (Optimized)']['confusion_matrix']['true_negatives'],
                 classifier.results['models']['SVM (Optimized)']['confusion_matrix']['false_positives']],
                [classifier.results['models']['SVM (Optimized)']['confusion_matrix']['false_negatives'],
                 classifier.results['models']['SVM (Optimized)']['confusion_matrix']['true_positives']]
            ]
            sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Reds', ax=ax3, cbar=False)
            ax3.set_title('Confusion Matrix - SVM (Optimized)')
            ax3.set_xlabel('Predicted')
            ax3.set_ylabel('Actual')
            ax3.set_xticklabels(['Benign', 'Malignant'])
            ax3.set_yticklabels(['Benign', 'Malignant'], rotation=0)
            
            # 4. ROC Curves
            ax4 = axes[1, 1]
            y_true = classifier.y_test.to_numpy()
            for model_name, model in models.items():
                y_score = classifier.predict_proba(model, classifier.X_test)
                fpr, tpr, _ = roc_curve(y_true, y_score)
                auc = roc_auc_score(y_true, y_score)
                ax4.plot(fpr, tpr, label=f'{model_name} (AUC={auc:.3f})', linewidth=2)
            
            ax4.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
            ax4.set_xlabel('False Positive Rate')
            ax4.set_ylabel('True Positive Rate')
            ax4.set_title('ROC Curves Comparison')
            ax4.legend()
            ax4.grid(alpha=0.3)
            
            plt.tight_layout()
            
            st.pyplot(fig)
            
            st.markdown("---")

            # Single patient demo
            st.header("🧪 Single-patient risk demo")
            st.warning(
                "Demo only. Not medical advice. Real clinical models need governance and prospective validation.",
                icon="⚠️",
            )

            model_choice = st.radio(
                "Model",
                options=["SVM (Optimized)", "Logistic Regression"],
                horizontal=True,
            )

            defaults = classifier.X_train.median(numeric_only=True).to_dict()
            top = classifier.results.get('interpretability', {}).get('lr_top_positive_coefficients', {})
            top_feats = list(top.keys()) if top else (classifier.feature_names[:10] if classifier.feature_names else [])

            with st.form("patient_form"):
                st.caption("Values default to training median.")
                patient_values = {}
                cols = st.columns(2)
                for i, feat in enumerate(top_feats):
                    col = cols[i % 2]
                    patient_values[feat] = col.number_input(
                        feat,
                        value=float(defaults.get(feat, 0.0)),
                        format="%.6f",
                    )
                submit = st.form_submit_button("Predict risk")

            if submit:
                full = {f: float(defaults.get(f, 0.0)) for f in (classifier.feature_names or [])}
                full.update({k: float(v) for k, v in patient_values.items()})
                pred = classifier.predict_patient(full, model_name=model_choice)
                st.metric("Predicted class", pred["predicted_class"].title())
                st.metric("Malignant probability", f"{pred['prob_malignant']:.3f}")
                st.caption(f"Threshold used: {pred['threshold']:.3f}")
            
            # Detailed Results
            st.header("📋 Detailed Results")
            
            with st.expander("📄 View Full Results JSON"):
                st.json(classifier.results)
            
            # Download results
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="📥 Download Results (JSON)",
                    data=json.dumps(classifier.results, indent=4),
                    file_name="cancer_classification_results.json",
                    mime="application/json"
                )
            
            with col2:
                # Save figure to bytes
                import io
                img_bytes = io.BytesIO()
                fig.savefig(img_bytes, format='png', dpi=300, bbox_inches='tight')
                img_bytes.seek(0)
                
                st.download_button(
                    label="📊 Download Visualization (PNG)",
                    data=img_bytes,
                    file_name="cancer_classification_visualization.png",
                    mime="image/png"
                )
            
            st.markdown("---")
            
            # Summary
            st.header("🎯 Summary & Recommendations")
            
            lr_acc = classifier.results['models']['Logistic Regression']['accuracy']
            svm_acc = classifier.results['models']['SVM (Optimized)']['accuracy']
            improvement = (svm_acc - lr_acc) * 100
            
            best_model = 'SVM (Optimized)' if svm_acc > lr_acc else 'Logistic Regression'
            
            st.info(f"""
            **Best Performing Model: {best_model}**
            
            - **Logistic Regression Accuracy**: {lr_acc:.4f} ({lr_acc*100:.2f}%)
            - **SVM (Optimized) Accuracy**: {svm_acc:.4f} ({svm_acc*100:.2f}%)
            - **Improvement**: {improvement:+.2f}%
            
            The {best_model} model shows superior performance on this dataset.
            """)
            
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            st.write("Please ensure all required packages are installed:")
            st.code("pip install -r requirements.txt", language="bash")

else:
    # Initial page content
    st.info("👈 Click the **Run Analysis** button in the sidebar to start the classification analysis.")
    
    st.markdown("""
    ## 📌 About This Application
    
    This application performs comprehensive machine learning analysis on the Wisconsin Breast Cancer dataset.
    
    ### Features:
    - 🔄 **Data Loading & Exploration**: Load and analyze the breast cancer dataset
    - 🔧 **Preprocessing**: Feature scaling and train-test splitting
    - 🤖 **Logistic Regression**: Baseline model for comparison
    - ⚡ **SVM with GridSearchCV**: Optimized Support Vector Machine
    - 📊 **Comprehensive Evaluation**: Multiple metrics and visualizations
    - 📥 **Results Export**: Download results and visualizations
    
    ### Dataset Information:
    - **Samples**: 569 breast cancer cases
    - **Features**: 30 numerical features
    - **Target**: Malignant (M) or Benign (B)
    
    ### Models Compared:
    1. **Logistic Regression** - Fast, interpretable baseline
    2. **Support Vector Machine (SVM)** - Optimized with GridSearchCV
    
    """)

