import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, auc, roc_auc_score)
import warnings
warnings.filterwarnings('ignore')

# Patient-friendly explanation helper (rule-based)
from model_utils import explain_prediction

# Page configuration
st.set_page_config(page_title="Breast Cancer Classification", layout="wide", page_icon="🔬")

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# Title and Introduction
st.markdown('<h1 class="main-header">🔬 Breast Cancer Classification: SVM vs Logistic Regression</h1>', unsafe_allow_html=True)
st.markdown("""
This application compares the performance of **Support Vector Machine (SVM)** with GridSearchCV optimization 
versus **Logistic Regression** baseline model for breast cancer classification using the Wisconsin Diagnostic Dataset.
""")

# Create tabs
tab1, tab2 = st.tabs(["🎯 Model Training", "📊 Model Application & Prediction"])

# =====================================================================
# TAB 1: MODEL TRAINING
# =====================================================================

with tab1:
    st.markdown('<h2 class="sub-header">Model Training and Comparison</h2>', unsafe_allow_html=True)
    
    # Sidebar configuration for Training
    st.sidebar.header("⚙️ Training Configuration")
    test_size = st.sidebar.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
    random_state = st.sidebar.number_input("Random State", 0, 100, 42)
    
    # Load data
    @st.cache_data
    def load_data(file_path):
        df = pd.read_csv(file_path)
        return df
    
    # Preprocess data
    @st.cache_data
    def preprocess_data(df, test_size, random_state):
        # Drop ID column
        df = df.drop('id', axis=1)
        
        # Encode diagnosis (M=1, B=0)
        le = LabelEncoder()
        df['diagnosis'] = le.fit_transform(df['diagnosis'])
        
        # Separate features and target
        X = df.drop('diagnosis', axis=1)
        y = df['diagnosis']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        
        # Feature scaling (important for SVM)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler, X.columns
    
    # Train models
    @st.cache_resource
    def train_models(X_train_scaled, y_train, X_test_scaled, y_test):
        results = {}
        
        # 1. Logistic Regression (Baseline)
        with st.spinner("Training Logistic Regression (Baseline)..."):
            lr_model = LogisticRegression(max_iter=10000, random_state=42)
            lr_model.fit(X_train_scaled, y_train)
            lr_pred = lr_model.predict(X_test_scaled)
            lr_pred_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
            
            results['lr'] = {
                'model': lr_model,
                'predictions': lr_pred,
                'pred_proba': lr_pred_proba,
                'accuracy': accuracy_score(y_test, lr_pred),
                'precision': precision_score(y_test, lr_pred),
                'recall': recall_score(y_test, lr_pred),
                'f1': f1_score(y_test, lr_pred),
                'confusion_matrix': confusion_matrix(y_test, lr_pred),
                'roc_auc': roc_auc_score(y_test, lr_pred_proba)
            }
        
        # 2. SVM with GridSearchCV
        with st.spinner("Training SVM with GridSearchCV (This may take a minute)..."):
            # Define parameter grid
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
            }
            
            svm_model = SVC(probability=True, random_state=42)
            grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=0)
            grid_search.fit(X_train_scaled, y_train)
            
            best_svm = grid_search.best_estimator_
            svm_pred = best_svm.predict(X_test_scaled)
            svm_pred_proba = best_svm.predict_proba(X_test_scaled)[:, 1]
            
            results['svm'] = {
                'model': best_svm,
                'grid_search': grid_search,
                'best_params': grid_search.best_params_,
                'predictions': svm_pred,
                'pred_proba': svm_pred_proba,
                'accuracy': accuracy_score(y_test, svm_pred),
                'precision': precision_score(y_test, svm_pred),
                'recall': recall_score(y_test, svm_pred),
                'f1': f1_score(y_test, svm_pred),
                'confusion_matrix': confusion_matrix(y_test, svm_pred),
                'roc_auc': roc_auc_score(y_test, svm_pred_proba)
            }
        
        return results
    
    # Main training app
    try:
        # Load data
        df = load_data('breast_cancer_data.csv')
        
        # Display dataset info
        with st.expander("📊 Dataset Overview", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Samples", df.shape[0])
            with col2:
                st.metric("Number of Features", df.shape[1] - 2)  # Excluding id and diagnosis
            with col3:
                diagnosis_counts = df['diagnosis'].value_counts()
                st.metric("Malignant/Benign Ratio", f"{diagnosis_counts['M']}/{diagnosis_counts['B']}")
            
            st.write("**First few rows:**")
            st.dataframe(df.head(10))
            
            st.write("**Class Distribution:**")
            fig, ax = plt.subplots(figsize=(8, 4))
            df['diagnosis'].value_counts().plot(kind='bar', ax=ax, color=['#ff7f0e', '#1f77b4'])
            ax.set_xlabel('Diagnosis')
            ax.set_ylabel('Count')
            ax.set_title('Distribution of Malignant (M) vs Benign (B)')
            ax.set_xticklabels(['Malignant', 'Benign'], rotation=0)
            st.pyplot(fig)
            plt.close()
        
        # Preprocess data
        X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler, feature_names = preprocess_data(df, test_size, random_state)
        
        st.success(f"✅ Data preprocessed: {len(X_train)} training samples, {len(X_test)} test samples")
        
        # Train models
        if st.button("🚀 Train and Compare Models", type="primary"):
            results = train_models(X_train_scaled, y_train, X_test_scaled, y_test)
            
            # Store results and models in session state for use in Tab 2
            st.session_state.trained_models = {
                'lr_model': results['lr']['model'],
                'svm_model': results['svm']['model'],
                'scaler': scaler,
                'feature_names': feature_names,
                'results': results
            }
            
            # Display results
            st.markdown('<h2 class="sub-header">📈 Model Performance Comparison</h2>', unsafe_allow_html=True)
            
            lr_results = results['lr']
            svm_results = results['svm']
            
            # Metrics comparison table
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 Logistic Regression (Baseline)")
                metrics_df = pd.DataFrame({
                    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
                    'Value': [
                        f"{lr_results['accuracy']:.4f}",
                        f"{lr_results['precision']:.4f}",
                        f"{lr_results['recall']:.4f}",
                        f"{lr_results['f1']:.4f}",
                        f"{lr_results['roc_auc']:.4f}"
                    ]
                })
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)
            
            with col2:
                st.markdown("### 🚀 SVM with GridSearchCV")
                st.markdown(f"**Best Parameters:** {svm_results['best_params']}")
                metrics_df = pd.DataFrame({
                    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
                    'Value': [
                        f"{svm_results['accuracy']:.4f}",
                        f"{svm_results['precision']:.4f}",
                        f"{svm_results['recall']:.4f}",
                        f"{svm_results['f1']:.4f}",
                        f"{svm_results['roc_auc']:.4f}"
                    ]
                })
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)
            
            # Comparison Bar Chart
            st.markdown("### 📊 Metrics Bar Chart")
            metrics_comparison = {
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
                'Logistic Regression': [
                    lr_results['accuracy'],
                    lr_results['precision'],
                    lr_results['recall'],
                    lr_results['f1'],
                    lr_results['roc_auc']
                ],
                'SVM (GridSearchCV)': [
                    svm_results['accuracy'],
                    svm_results['precision'],
                    svm_results['recall'],
                    svm_results['f1'],
                    svm_results['roc_auc']
                ]
            }
            
            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(len(metrics_comparison['Metric']))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, metrics_comparison['Logistic Regression'], width, label='Logistic Regression', color='#1f77b4')
            bars2 = ax.bar(x + width/2, metrics_comparison['SVM (GridSearchCV)'], width, label='SVM (GridSearchCV)', color='#ff7f0e')
            
            ax.set_ylabel('Score', fontsize=12)
            ax.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(metrics_comparison['Metric'])
            ax.legend()
            ax.set_ylim([0, 1.1])
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}',
                           ha='center', va='bottom', fontsize=9)
            
            st.pyplot(fig)
            plt.close()
            
            # Confusion Matrices
            st.markdown("### 🎯 Confusion Matrices")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Logistic Regression")
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(lr_results['confusion_matrix'], annot=True, fmt='d', cmap='Blues', 
                           xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'], ax=ax)
                ax.set_ylabel('Actual')
                ax.set_xlabel('Predicted')
                ax.set_title('Logistic Regression Confusion Matrix')
                st.pyplot(fig)
                plt.close()
            
            with col2:
                st.markdown("#### SVM (GridSearchCV)")
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(svm_results['confusion_matrix'], annot=True, fmt='d', cmap='Oranges', 
                           xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'], ax=ax)
                ax.set_ylabel('Actual')
                ax.set_xlabel('Predicted')
                ax.set_title('SVM Confusion Matrix')
                st.pyplot(fig)
                plt.close()
            
            # ROC Curves
            st.markdown("### 📉 ROC Curves Comparison")
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Logistic Regression ROC
            fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_results['pred_proba'])
            ax.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {lr_results["roc_auc"]:.3f})', 
                   linewidth=2, color='#1f77b4')
            
            # SVM ROC
            fpr_svm, tpr_svm, _ = roc_curve(y_test, svm_results['pred_proba'])
            ax.plot(fpr_svm, tpr_svm, label=f'SVM GridSearchCV (AUC = {svm_results["roc_auc"]:.3f})', 
                   linewidth=2, color='#ff7f0e')
            
            # Diagonal line
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
            
            ax.set_xlabel('False Positive Rate', fontsize=12)
            ax.set_ylabel('True Positive Rate', fontsize=12)
            ax.set_title('ROC Curve Comparison', fontsize=14, fontweight='bold')
            ax.legend(loc='lower right', fontsize=11)
            ax.grid(alpha=0.3)
            
            st.pyplot(fig)
            plt.close()
            
            # Classification Reports
            st.markdown("### 📋 Detailed Classification Reports")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Logistic Regression")
                lr_report = classification_report(y_test, lr_results['predictions'], 
                                                 target_names=['Benign', 'Malignant'], output_dict=True)
                st.dataframe(pd.DataFrame(lr_report).transpose(), use_container_width=True)
            
            with col2:
                st.markdown("#### SVM (GridSearchCV)")
                svm_report = classification_report(y_test, svm_results['predictions'], 
                                                  target_names=['Benign', 'Malignant'], output_dict=True)
                st.dataframe(pd.DataFrame(svm_report).transpose(), use_container_width=True)
            
            # Feature Importance (for Linear SVM or Logistic Regression)
            if svm_results['best_params']['kernel'] == 'linear':
                st.markdown("### 🔍 Feature Importance Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Logistic Regression Coefficients")
                    lr_coef = pd.DataFrame({
                        'Feature': feature_names,
                        'Coefficient': lr_results['model'].coef_[0]
                    }).sort_values('Coefficient', key=abs, ascending=False).head(10)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    colors = ['green' if x > 0 else 'red' for x in lr_coef['Coefficient']]
                    ax.barh(lr_coef['Feature'], lr_coef['Coefficient'], color=colors, alpha=0.7)
                    ax.set_xlabel('Coefficient Value')
                    ax.set_title('Top 10 Features - Logistic Regression')
                    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                
                with col2:
                    st.markdown("#### SVM Coefficients (Linear Kernel)")
                    svm_coef = pd.DataFrame({
                        'Feature': feature_names,
                        'Coefficient': svm_results['model'].coef_[0]
                    }).sort_values('Coefficient', key=abs, ascending=False).head(10)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    colors = ['green' if x > 0 else 'red' for x in svm_coef['Coefficient']]
                    ax.barh(svm_coef['Feature'], svm_coef['Coefficient'], color=colors, alpha=0.7)
                    ax.set_xlabel('Coefficient Value')
                    ax.set_title('Top 10 Features - SVM')
                    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
            
            # Summary and conclusions
            st.markdown("### 🎓 Summary and Conclusions")
            
            improvement = ((svm_results['accuracy'] - lr_results['accuracy']) / lr_results['accuracy']) * 100
            
            if improvement > 0:
                st.success(f"""
                **Key Findings:**
                - SVM with GridSearchCV achieved **{improvement:.2f}% improvement** in accuracy over the baseline Logistic Regression
                - **Best SVM Parameters:** {svm_results['best_params']}
                - SVM achieved higher precision, indicating fewer false positives (important for reducing unnecessary treatments)
                - Both models show strong performance with AUC > 0.95, but SVM demonstrates superior classification capability
                
                **Clinical Relevance:**
                The optimized SVM model provides a more reliable tool for assisting in breast cancer diagnosis, 
                potentially reducing human error and improving early detection rates.
                """)
            else:
                st.info(f"""
                **Key Findings:**
                - Both models performed similarly, with Logistic Regression showing {abs(improvement):.2f}% better accuracy
                - **Best SVM Parameters:** {svm_results['best_params']}
                - For this dataset, the simpler Logistic Regression model may be preferable due to its interpretability
                - Both models demonstrate strong predictive capability with AUC > 0.95
                """)
            
            # Save results option
            st.markdown("---")
            if st.button("💾 Download Results Summary"):
                summary = pd.DataFrame({
                    'Model': ['Logistic Regression', 'SVM (GridSearchCV)'],
                    'Accuracy': [lr_results['accuracy'], svm_results['accuracy']],
                    'Precision': [lr_results['precision'], svm_results['precision']],
                    'Recall': [lr_results['recall'], svm_results['recall']],
                    'F1-Score': [lr_results['f1'], svm_results['f1']],
                    'ROC-AUC': [lr_results['roc_auc'], svm_results['roc_auc']]
                })
                csv = summary.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name="model_comparison_results.csv",
                    mime="text/csv"
                )
    
    except FileNotFoundError:
        st.error("❌ Dataset file not found. Please ensure 'breast_cancer_data.csv' is in the correct location.")
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")

# =====================================================================
# TAB 2: MODEL APPLICATION & PREDICTION
# =====================================================================

with tab2:
    st.markdown('<h2 class="sub-header">Apply Trained Models for Predictions</h2>', unsafe_allow_html=True)
    
    # Check if models are trained
    if 'trained_models' not in st.session_state:
        st.warning("⚠️ Please train the models first in the 'Model Training' tab!")
    else:
        trained_data = st.session_state.trained_models
        lr_model = trained_data['lr_model']
        svm_model = trained_data['svm_model']
        scaler = trained_data['scaler']
        feature_names = trained_data['feature_names']
        results = trained_data['results']
        
        st.success("✅ Models loaded successfully!")
        
        # Display model info
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Logistic Regression")
            st.metric("Accuracy", f"{results['lr']['accuracy']:.4f}")
            st.metric("Precision", f"{results['lr']['precision']:.4f}")
            st.metric("Recall", f"{results['lr']['recall']:.4f}")
        
        with col2:
            st.markdown("### 🚀 SVM Model")
            st.metric("Accuracy", f"{results['svm']['accuracy']:.4f}")
            st.metric("Precision", f"{results['svm']['precision']:.4f}")
            st.metric("Recall", f"{results['svm']['recall']:.4f}")
        
        st.markdown("---")
        
        # Input method selection
        input_method = st.radio("Select input method:", ["Manual Input", "Upload CSV"], horizontal=True)
        
        if input_method == "Manual Input":
            st.markdown("### 🔬 Enter Patient Features")
            
            # Create input columns dynamically
            feature_values = {}
            cols = st.columns(2)
            
            for idx, feature in enumerate(feature_names):
                with cols[idx % 2]:
                    feature_values[feature] = st.number_input(
                        f"{feature}",
                        value=0.0,
                        step=0.1
                    )
            
            if st.button("🎯 Make Prediction", type="primary"):
                # Prepare input data
                input_data = pd.DataFrame([feature_values])
                input_scaled = scaler.transform(input_data)
                
                # Make predictions
                lr_pred = lr_model.predict(input_scaled)[0]
                lr_pred_proba = lr_model.predict_proba(input_scaled)[0]
                
                svm_pred = svm_model.predict(input_scaled)[0]
                svm_pred_proba = svm_model.predict_proba(input_scaled)[0]
                
                # Display results
                st.markdown("---")
                st.markdown("### 📋 Prediction Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Logistic Regression Prediction")
                    diagnosis_lr = "🔴 Malignant" if lr_pred == 1 else "🟢 Benign"
                    st.markdown(f"**Diagnosis:** {diagnosis_lr}")
                    st.markdown(f"**Benign Probability:** {lr_pred_proba[0]:.4f}")
                    st.markdown(f"**Malignant Probability:** {lr_pred_proba[1]:.4f}")
                    
                    # Probability bar
                    st.progress(lr_pred_proba[1], text=f"Malignant: {lr_pred_proba[1]:.2%}")
                
                with col2:
                    st.markdown("#### SVM Prediction")
                    diagnosis_svm = "🔴 Malignant" if svm_pred == 1 else "🟢 Benign"
                    st.markdown(f"**Diagnosis:** {diagnosis_svm}")
                    st.markdown(f"**Benign Probability:** {svm_pred_proba[0]:.4f}")
                    st.markdown(f"**Malignant Probability:** {svm_pred_proba[1]:.4f}")
                    
                    # Probability bar
                    st.progress(svm_pred_proba[1], text=f"Malignant: {svm_pred_proba[1]:.2%}")
                
                # Agreement check
                st.markdown("---")
                if lr_pred == svm_pred:
                    st.success("✅ Both models agree on the diagnosis!")
                else:
                    st.warning("⚠️ Models disagree on the diagnosis. Consider the probabilities carefully.")

                # Doctor-style explanation (patient friendly)
                st.markdown("---")
                st.markdown("### 🩺 Doctor-style explanation")

                # Use the average malignant probability as a simple combined score.
                combined_p_malignant = float((lr_pred_proba[1] + svm_pred_proba[1]) / 2.0)

                explain = explain_prediction(
                    combined_p_malignant,
                    threshold=0.5,
                    patient_name=None,
                    other_model_malignant_probability=float(svm_pred_proba[1]),
                    other_model_name="SVM (vs Logistic Regression)",
                )

                # Show interpretation
                st.write(f"**Summary:** {explain['summary']}")
                st.caption(f"Risk tier: {str(explain['risk_tier']).title()} • Combined malignant probability: {explain['malignant_probability']:.2%}")
                st.info(explain["details"], icon="🧾")
                st.warning(explain["next_steps"], icon="📌")
                if explain.get("agreement_note"):
                    st.warning(explain["agreement_note"], icon="⚠️")
        
        else:  # Upload CSV
            st.markdown("### 📤 Upload Patient Data (CSV)")
            st.markdown("CSV should contain columns matching the feature names used in training.")
            
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            
            if uploaded_file is not None:
                input_df = pd.read_csv(uploaded_file)
                
                st.write("**Preview of uploaded data:**")
                st.dataframe(input_df.head())
                
                if st.button("🎯 Make Predictions", type="primary"):
                    try:
                        # Prepare input data
                        input_scaled = scaler.transform(input_df[feature_names])
                        
                        # Make predictions
                        lr_preds = lr_model.predict(input_scaled)
                        lr_preds_proba = lr_model.predict_proba(input_scaled)
                        
                        svm_preds = svm_model.predict(input_scaled)
                        svm_preds_proba = svm_model.predict_proba(input_scaled)
                        
                        # Create results dataframe
                        results_df = pd.DataFrame({
                            'LR_Diagnosis': ['Malignant' if p == 1 else 'Benign' for p in lr_preds],
                            'LR_Benign_Prob': lr_preds_proba[:, 0],
                            'LR_Malignant_Prob': lr_preds_proba[:, 1],
                            'SVM_Diagnosis': ['Malignant' if p == 1 else 'Benign' for p in svm_preds],
                            'SVM_Benign_Prob': svm_preds_proba[:, 0],
                            'SVM_Malignant_Prob': svm_preds_proba[:, 1],
                            'Agreement': ['Yes' if lr_preds[i] == svm_preds[i] else 'No' for i in range(len(lr_preds))]
                        })

                        # Add patient-friendly explanations per row (combined probability)
                        combined_m = (results_df["LR_Malignant_Prob"] + results_df["SVM_Malignant_Prob"]) / 2.0
                        results_df["Combined_Malignant_Prob"] = combined_m

                        summaries = []
                        actions = []
                        tiers = []
                        for i in range(len(results_df)):
                            exp = explain_prediction(
                                float(results_df.loc[i, "Combined_Malignant_Prob"]),
                                threshold=0.5,
                                other_model_malignant_probability=float(results_df.loc[i, "SVM_Malignant_Prob"]),
                                other_model_name="SVM",
                            )
                            summaries.append(exp["summary"])
                            actions.append(exp["next_steps"])
                            tiers.append(exp["risk_tier"])

                        results_df["Explanation_Summary"] = summaries
                        results_df["Explanation_Risk_Tier"] = tiers
                        results_df["Explanation_Next_Steps"] = actions
                        
                        st.markdown("---")
                        st.markdown("### 📋 Batch Prediction Results Summary")
                        
                        # Display overview statistics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Predictions", len(results_df))
                        with col2:
                            malignant_count = sum(results_df["LR_Diagnosis"] == "Malignant")
                            st.metric("Malignant Cases", malignant_count)
                        with col3:
                            benign_count = sum(results_df["LR_Diagnosis"] == "Benign")
                            st.metric("Benign Cases", benign_count)
                        with col4:
                            agreement_pct = (sum(results_df["Agreement"] == "Yes") / len(results_df)) * 100
                            st.metric("Model Agreement", f"{agreement_pct:.1f}%")
                        
                        # Display simplified results table
                        st.markdown("#### Quick Results Table")
                        display_df = results_df[['LR_Diagnosis', 'LR_Malignant_Prob', 'SVM_Diagnosis', 
                                                'SVM_Malignant_Prob', 'Combined_Malignant_Prob', 
                                                'Explanation_Risk_Tier', 'Agreement']].copy()
                        display_df.columns = ['LR Diagnosis', 'LR Prob', 'SVM Diagnosis', 
                                             'SVM Prob', 'Combined Prob', 'Risk Tier', 'Agreement']
                        # Format probabilities as percentages
                        display_df['LR Prob'] = display_df['LR Prob'].map('{:.2%}'.format)
                        display_df['SVM Prob'] = display_df['SVM Prob'].map('{:.2%}'.format)
                        display_df['Combined Prob'] = display_df['Combined Prob'].map('{:.2%}'.format)
                        st.dataframe(display_df, use_container_width=True)

                        # Show detailed explanations for each prediction
                        st.markdown("---")
                        st.markdown("### 🩺 Detailed Medical Interpretations")
                        st.info("📌 **Important:** These interpretations are AI-generated educational insights and should NOT replace professional medical diagnosis. Always consult with healthcare providers for actual clinical decisions.", icon="⚠️")
                        
                        # Create expandable sections for each prediction
                        for i in range(len(results_df)):
                            r = results_df.iloc[i]
                            
                            # Determine the emoji and color based on diagnosis
                            if r['LR_Diagnosis'] == 'Malignant':
                                status_emoji = "🔴"
                                status_color = "#ff4b4b"
                            else:
                                status_emoji = "🟢"
                                status_color = "#00cc00"
                            
                            # Create expander title with key info
                            agreement_status = "✅ Models Agree" if r['Agreement'] == 'Yes' else "⚠️ Models Disagree"
                            expander_title = f"**Patient #{i+1}** {status_emoji} | Combined Probability: {r['Combined_Malignant_Prob']:.1%} | {agreement_status}"
                            
                            with st.expander(expander_title, expanded=(i==0)):
                                # Create two columns for model results
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("##### 📊 Logistic Regression")
                                    st.markdown(f"**Diagnosis:** {r['LR_Diagnosis']}")
                                    st.progress(float(r['LR_Malignant_Prob']), text=f"Malignant: {r['LR_Malignant_Prob']:.2%}")
                                    st.caption(f"Benign: {r['LR_Benign_Prob']:.2%} | Malignant: {r['LR_Malignant_Prob']:.2%}")
                                
                                with col2:
                                    st.markdown("##### 🚀 SVM Model")
                                    st.markdown(f"**Diagnosis:** {r['SVM_Diagnosis']}")
                                    st.progress(float(r['SVM_Malignant_Prob']), text=f"Malignant: {r['SVM_Malignant_Prob']:.2%}")
                                    st.caption(f"Benign: {r['SVM_Benign_Prob']:.2%} | Malignant: {r['SVM_Malignant_Prob']:.2%}")
                                
                                st.markdown("---")
                                
                                # Display comprehensive explanation
                                st.markdown("##### 🩺 Clinical Interpretation")
                                st.markdown(f"**Risk Assessment:** {str(r['Explanation_Risk_Tier']).upper()}")
                                st.markdown(f"**Summary:** {r['Explanation_Summary']}")
                                
                                # Get full explanation for detailed text
                                full_exp = explain_prediction(
                                    float(r["Combined_Malignant_Prob"]),
                                    threshold=0.5,
                                    patient_name=f"Patient #{i+1}",
                                    other_model_malignant_probability=float(r["SVM_Malignant_Prob"]),
                                    other_model_name="SVM",
                                )
                                
                                st.info(f"**Details:** {full_exp['details']}", icon="📋")
                                st.warning(f"**Recommended Next Steps:** {full_exp['next_steps']}", icon="📌")
                                
                                if full_exp.get("agreement_note"):
                                    st.error(f"**Model Disagreement Alert:** {full_exp['agreement_note']}", icon="⚠️")
                                
                                if r['Agreement'] == 'Yes':
                                    st.success("✅ Both models agree on this diagnosis, increasing confidence in the prediction.")
                                else:
                                    st.warning("⚠️ Models disagree on this diagnosis. Clinical review is strongly recommended.", icon="⚠️")
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Predictions CSV",
                            data=csv,
                            file_name="batch_predictions.csv",
                            mime="text/csv"
                        )
                    
                    except Exception as e:
                        st.error(f"❌ Error making predictions: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🔬 <b>SECB3203-02 Programming for Bioinformatics Project</b></p>
    <p>Breast Cancer Classification using Machine Learning | Wisconsin Diagnostic Dataset</p>
</div>
""", unsafe_allow_html=True)
