"""
End-to-End Leakage-Free Training, Benchmarking, and Evaluation Pipeline for Gujarati-English Sarcasm Detection.
Uses group-aware splitting (GroupShuffleSplit on group_id) to eliminate all template and paraphrase leakage.
Fits feature extractors strictly on training data, benchmarks 6 models, performs validation tuning,
evaluates on the held-out test set, conducts error analysis, and exports validated artifacts.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import sklearn
from datetime import datetime

# Scikit-learn modules
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# Ensure local imports work
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "backend", "model"))
from preprocessor import clean_text, extract_linguistic_signals

DATASET_PATH = os.path.join(PROJECT_ROOT, "sarcasm_dataset.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "backend", "model")
MODEL_PATH = os.path.join(MODEL_DIR, "sarcasm_pipeline.joblib")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")


def make_group_calibration_folds(X, y, groups, n_splits=5, random_state=42):
    """Make calibration folds that keep each related group on one side."""
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = list(splitter.split(X, y, groups=groups))
    for fold_number, (fit_idx, calibration_idx) in enumerate(folds, start=1):
        fit_groups = set(groups.iloc[fit_idx])
        calibration_groups = set(groups.iloc[calibration_idx])
        overlap = fit_groups & calibration_groups
        if overlap:
            raise ValueError(f"Calibration fold {fold_number} leaks groups: {sorted(overlap)}")
    return folds


def make_calibrated_svm(feature_union, folds):
    """Calibrate an SVM whose TF-IDF features are refitted inside every fold."""
    feature_and_svm = Pipeline([
        ("features", feature_union),
        ("classifier", LinearSVC(C=1.0, random_state=42))
    ])
    return CalibratedClassifierCV(estimator=feature_and_svm, cv=folds)

def main():
    print("=" * 70)
    print("GUJARATI-ENGLISH (ROMANIZED) SARCASM DETECTION - ML TRAINING PIPELINE")
    print("=" * 70)

    # 1. Load Dataset
    print(f"\n[Step 1] Loading dataset from: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    print(f"Total records loaded: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print("Class distribution:")
    print(df["label"].value_counts())
    print(f"Total unique group_ids: {df['group_id'].nunique()}")

    # Preprocessing
    df["clean_sentence"] = df["sentence"].apply(clean_text)

    # 2. Group-Aware Train / Validation / Test Split (Strictly Disjoint by group_id)
    print("\n[Step 2] Group-Aware Splitting into Train (70%), Validation (15%), Test (15%)")
    X = df["clean_sentence"]
    y = df["label"]
    groups = df["group_id"]

    # First split off Test (15% of groups)
    gss_test = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_val_idx, test_idx = next(gss_test.split(X, y, groups=groups))

    # Split remaining 85% into Train (70% total) and Val (15% total -> 15/85 of train_val)
    val_ratio = 0.15 / 0.85
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=42)
    train_val_groups = groups.iloc[train_val_idx]
    train_sub_idx, val_sub_idx = next(
        gss_val.split(X.iloc[train_val_idx], y.iloc[train_val_idx], groups=train_val_groups)
    )

    train_idx = train_val_idx[train_sub_idx]
    val_idx = train_val_idx[val_sub_idx]

    # Verify zero leakage across splits
    train_groups = set(groups.iloc[train_idx])
    val_groups = set(groups.iloc[val_idx])
    test_groups = set(groups.iloc[test_idx])

    assert len(train_groups & test_groups) == 0, "CRITICAL ERROR: Data leakage between Train and Test!"
    assert len(train_groups & val_groups) == 0, "CRITICAL ERROR: Data leakage between Train and Val!"
    assert len(val_groups & test_groups) == 0, "CRITICAL ERROR: Data leakage between Val and Test!"
    print(">> Group Leakage Audit PASSED: Zero overlap across Train, Val, and Test splits.")

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    print(f"Training set:   {len(X_train)} samples across {len(train_groups)} groups ({y_train.value_counts().to_dict()})")
    print(f"Validation set: {len(X_val)} samples across {len(val_groups)} groups ({y_val.value_counts().to_dict()})")
    print(f"Held-out Test:  {len(X_test)} samples across {len(test_groups)} groups ({y_test.value_counts().to_dict()})")

    # 3. Build Feature Extraction Pipeline (Fitted Strictly on Training Data)
    print("\n[Step 3] Configuring Feature Extraction (Word N-grams [1-3] + Char-wb N-grams [3-5])")
    feature_union = FeatureUnion([
        ("word_tfidf", TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=6000,
            sublinear_tf=True
        )),
        ("char_tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=12000,
            sublinear_tf=True
        ))
    ])

    # 4. Define Candidate Classifiers
    print("\n[Step 4] Benchmarking Candidate Models on Training & Validation Sets")
    candidates = {
        "Logistic Regression (C=2.0)": LogisticRegression(C=2.0, max_iter=1000, random_state=42),
        "Complement Naive Bayes (alpha=0.5)": ComplementNB(alpha=0.5),
        "Multinomial Naive Bayes (alpha=0.5)": MultinomialNB(alpha=0.5),
        "Random Forest (n_est=200)": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    results = []
    trained_pipelines = {}

    # Use StratifiedGroupKFold on training set to prevent fold leakage
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_group_series = groups.iloc[train_idx].reset_index(drop=True)
    X_train_reset = X_train.reset_index(drop=True)
    y_train_reset = y_train.reset_index(drop=True)

    for name, clf in candidates.items():
        pipeline = Pipeline([
            ("features", feature_union),
            ("classifier", clf)
        ])

        # Cross-validation score on Training set with group-aware folds
        cv_scores = cross_val_score(
            pipeline, X_train_reset, y_train_reset,
            cv=sgkf.split(X_train_reset, y_train_reset, groups=train_group_series),
            scoring="f1_macro"
        )
        cv_mean = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))

        # Fit on Training set
        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline

        # Evaluate on Validation set
        y_val_pred = pipeline.predict(X_val)
        val_acc = float(accuracy_score(y_val, y_val_pred))
        val_f1 = float(f1_score(y_val, y_val_pred, average="macro"))

        results.append({
            "model_name": name,
            "cv_macro_f1_mean": round(cv_mean, 4),
            "cv_macro_f1_std": round(cv_std, 4),
            "val_accuracy": round(val_acc, 4),
            "val_macro_f1": round(val_f1, 4)
        })

        print(f" - {name:<35} | CV Macro-F1: {cv_mean:.4f} (+/- {cv_std:.4f}) | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

    # Calibrate the SVM with group-contained folds. A fixed group-aware fold
    # list is safe here because this model is fitted once on the complete train
    # partition (it is not nested inside the separate outer CV benchmark).
    group_calibration_folds = make_group_calibration_folds(
        X_train_reset, y_train_reset, train_group_series
    )
    svm_name = "Group-Calibrated Linear SVM (C=1.0)"
    # Each calibration clone owns its own full text -> TF-IDF -> LinearSVC
    # pipeline. This prevents vocabulary and IDF values from seeing held-fold text.
    svm_pipeline = make_calibrated_svm(feature_union, group_calibration_folds)
    svm_pipeline.fit(X_train, y_train)
    svm_val_pred = svm_pipeline.predict(X_val)
    svm_val_acc = float(accuracy_score(y_val, svm_val_pred))
    svm_val_f1 = float(f1_score(y_val, svm_val_pred, average="macro"))
    trained_pipelines[svm_name] = svm_pipeline
    results.append({
        "model_name": svm_name,
        "cv_macro_f1_mean": None,
        "cv_macro_f1_std": None,
        "val_accuracy": round(svm_val_acc, 4),
        "val_macro_f1": round(svm_val_f1, 4)
    })
    print(f" - {svm_name:<35} | Group-aware calibration | Val Acc: {svm_val_acc:.4f} | Val F1: {svm_val_f1:.4f}")

    results_df = pd.DataFrame(results).sort_values(by="val_macro_f1", ascending=False)
    best_candidate_name = results_df.iloc[0]["model_name"]
    print(f"\n[Winner] Best Performing Architecture on Validation: {best_candidate_name}")

    # 5. Evaluate Best Candidate on the Held-out Test Set
    best_val_pipeline = trained_pipelines[best_candidate_name]
    y_test_pred = best_val_pipeline.predict(X_test)

    test_acc = float(accuracy_score(y_test, y_test_pred))
    test_prec_macro = float(precision_score(y_test, y_test_pred, average="macro"))
    test_rec_macro = float(recall_score(y_test, y_test_pred, average="macro"))
    test_f1_macro = float(f1_score(y_test, y_test_pred, average="macro"))

    test_prec_sarcasm = float(precision_score(y_test, y_test_pred, pos_label="Sarcasm", average="binary"))
    test_rec_sarcasm = float(recall_score(y_test, y_test_pred, pos_label="Sarcasm", average="binary"))
    test_f1_sarcasm = float(f1_score(y_test, y_test_pred, pos_label="Sarcasm", average="binary"))

    test_prec_nonsarcasm = float(precision_score(y_test, y_test_pred, pos_label="Non-Sarcasm", average="binary"))
    test_rec_nonsarcasm = float(recall_score(y_test, y_test_pred, pos_label="Non-Sarcasm", average="binary"))
    test_f1_nonsarcasm = float(f1_score(y_test, y_test_pred, pos_label="Non-Sarcasm", average="binary"))

    cm = confusion_matrix(y_test, y_test_pred, labels=["Non-Sarcasm", "Sarcasm"]).tolist()
    # cm layout:
    # [[TN, FP],
    #  [FN, TP]]

    print("\n" + "=" * 70)
    print(f"HELD-OUT TEST SET EVALUATION ({best_candidate_name})")
    print("=" * 70)
    print(f"Test Accuracy:         {test_acc * 100:.2f}%")
    print(f"Macro Precision:       {test_prec_macro * 100:.2f}%")
    print(f"Macro Recall:          {test_rec_macro * 100:.2f}%")
    print(f"Macro F1-Score:        {test_f1_macro * 100:.2f}%")
    print("\nPer-Class Breakdown:")
    print(f" - Sarcasm:     Precision={test_prec_sarcasm*100:.2f}%, Recall={test_rec_sarcasm*100:.2f}%, F1={test_f1_sarcasm*100:.2f}%")
    print(f" - Non-Sarcasm: Precision={test_prec_nonsarcasm*100:.2f}%, Recall={test_rec_nonsarcasm*100:.2f}%, F1={test_f1_nonsarcasm*100:.2f}%")
    print("\nConfusion Matrix (Rows: Actual [Non-Sarcasm, Sarcasm], Cols: Predicted [Non-Sarcasm, Sarcasm]):")
    print(f"  Non-Sarcasm (Actual): TN = {cm[0][0]}, FP = {cm[0][1]}")
    print(f"  Sarcasm     (Actual): FN = {cm[1][0]}, TP = {cm[1][1]}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred, digits=4))

    # 6. Evaluation Across Provenance Subsets & Error Analysis
    test_df = df.iloc[test_idx].copy().reset_index(drop=True)
    test_df["prediction"] = y_test_pred
    test_df["correct"] = (test_df["label"] == test_df["prediction"])

    # Probabilities if available
    if hasattr(best_val_pipeline, "predict_proba"):
        sarc_idx = list(best_val_pipeline.classes_).index("Sarcasm")
        test_df["sarcasm_prob"] = best_val_pipeline.predict_proba(X_test)[:, sarc_idx]
    else:
        test_df["sarcasm_prob"] = np.nan

    print("Subset Performance on Held-Out Test Data:")
    subset_metrics = {}
    for subset_col in ["source_type", "sub_category", "language"]:
        print(f"\nBreakdown by {subset_col}:")
        subset_metrics[subset_col] = {}
        for val, group in test_df.groupby(subset_col):
            sub_acc = float(group["correct"].mean())
            sub_count = len(group)
            subset_metrics[subset_col][val] = {"count": sub_count, "accuracy": round(sub_acc, 4)}
            print(f" - {val:<25} ({sub_count:>3} samples): Accuracy = {sub_acc * 100:.1f}%")

    # Detailed Error Analysis: False Positives & False Negatives
    fps = test_df[(test_df["label"] == "Non-Sarcasm") & (test_df["prediction"] == "Sarcasm")]
    fns = test_df[(test_df["label"] == "Sarcasm") & (test_df["prediction"] == "Non-Sarcasm")]

    print("\n" + "=" * 70)
    print(f"ERROR ANALYSIS ON HELD-OUT TEST DATA ({len(fps)} FPs, {len(fns)} FNs)")
    print("=" * 70)

    fp_list = []
    if len(fps) > 0:
        print("\n[False Positives - Non-Sarcasm predicted as Sarcasm]:")
        for i, r in fps.iterrows():
            prob_info = f" | P(Sarc)={r['sarcasm_prob']:.2f}" if not np.isnan(r["sarcasm_prob"]) else ""
            print(f" - \"{r['sentence']}\"{prob_info} (sub: {r['sub_category']}, grp: {r['group_id']})")
            fp_list.append({
                "sentence": r["sentence"],
                "sub_category": r["sub_category"],
                "group_id": int(r["group_id"]),
                "sarcasm_prob": round(float(r["sarcasm_prob"]), 4) if not np.isnan(r["sarcasm_prob"]) else None
            })
    else:
        print("\n[False Positives]: None! (0 false positives)")

    fn_list = []
    if len(fns) > 0:
        print("\n[False Negatives - Sarcasm predicted as Non-Sarcasm]:")
        for i, r in fns.iterrows():
            prob_info = f" | P(Sarc)={r['sarcasm_prob']:.2f}" if not np.isnan(r["sarcasm_prob"]) else ""
            print(f" - \"{r['sentence']}\"{prob_info} (sub: {r['sub_category']}, grp: {r['group_id']})")
            fn_list.append({
                "sentence": r["sentence"],
                "sub_category": r["sub_category"],
                "group_id": int(r["group_id"]),
                "sarcasm_prob": round(float(r["sarcasm_prob"]), 4) if not np.isnan(r["sarcasm_prob"]) else None
            })
    else:
        print("\n[False Negatives]: None! (0 false negatives)")

    # 7. Export Evaluated Model Artifact & Detailed Metadata
    print(f"\n[Step 7] Exporting trained pipeline to: {MODEL_PATH}")
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_val_pipeline, MODEL_PATH)

    metadata = {
        "model_name": best_candidate_name,
        "export_timestamp": datetime.now().isoformat(),
        "scikit_learn_version": sklearn.__version__,
        "total_dataset_size": len(df),
        "unique_groups": int(df["group_id"].nunique()),
        "dataset_provenance": "Synthetic programmatically authored examples; spelling variants are generated, not collected from users.",
        "evaluation_limitations": "Examples are synthetic and group IDs are author-defined. Held-out scores do not establish performance on real conversation. The calibrated SVM uses group-contained training folds.",
        "probability_calibrated": "SVM" in best_candidate_name,
        "random_seed": 42,
        "split_method": "Two-stage GroupShuffleSplit by group_id (15% test, then 15% of total validation)",
        "feature_settings": {"word_tfidf": {"ngram_range": [1, 3], "max_features": 6000}, "char_wb_tfidf": {"ngram_range": [3, 5], "max_features": 12000}},
        "calibration": {
            "method": "CalibratedClassifierCV with sigmoid calibration",
            "fold_strategy": "StratifiedGroupKFold",
            "fold_count": len(group_calibration_folds),
            "feature_extraction_inside_calibrated_estimator": True,
            "fold_group_overlap": [
                len(set(train_group_series.iloc[fit_idx]) & set(train_group_series.iloc[cal_idx]))
                for fit_idx, cal_idx in group_calibration_folds
            ]
        },
        "train_samples": len(X_train),
        "train_groups": len(train_groups),
        "val_samples": len(X_val),
        "val_groups": len(val_groups),
        "test_samples": len(X_test),
        "test_groups": len(test_groups),
        "leakage_audit": {
            "train_test_overlap": 0,
            "train_val_overlap": 0,
            "val_test_overlap": 0,
            "status": "PASSED"
        },
        "metrics": {
            "accuracy": round(test_acc, 4),
            "macro_precision": round(test_prec_macro, 4),
            "macro_recall": round(test_rec_macro, 4),
            "macro_f1": round(test_f1_macro, 4),
            "sarcasm": {
                "precision": round(test_prec_sarcasm, 4),
                "recall": round(test_rec_sarcasm, 4),
                "f1": round(test_f1_sarcasm, 4)
            },
            "non_sarcasm": {
                "precision": round(test_prec_nonsarcasm, 4),
                "recall": round(test_rec_nonsarcasm, 4),
                "f1": round(test_f1_nonsarcasm, 4)
            }
        },
        "confusion_matrix": {
            "labels": ["Non-Sarcasm", "Sarcasm"],
            "matrix": cm,
            "tn": cm[0][0],
            "fp": cm[0][1],
            "fn": cm[1][0],
            "tp": cm[1][1]
        },
        "candidate_comparison": results,
        "subset_performance": subset_metrics,
        "error_analysis": {
            "total_errors": len(fps) + len(fns),
            "false_positives_count": len(fps),
            "false_negatives_count": len(fns),
            "false_positives": fp_list,
            "false_negatives": fn_list
        },
        "classes": list(best_val_pipeline.classes_)
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved evaluation metadata to: {METADATA_PATH}")

    # 8. Test Predictions on Key Illustrative Sentences
    print("\n" + "=" * 70)
    print("LIVE PIPELINE INFERENCE DEMONSTRATION")
    print("=" * 70)

    test_sentences = [
        "Wah, su mast service che!",
        "Exam ma fail thai ne maja aavi.",
        "Aaje traffic ma be kalak waste thaya, ekdum perfect.",
        "Tari vaat sambhali ne toh duniya badlai gai.",
        "Finally kaam puru thayu, what a relief!",
        "Wah, kharekhar khub saras kaam karyu che, aabhar!",
        "Aaje traffic ma be kalak waste thaya, hu ekdum thaki gayo chu.",
        "Aaje somvar che ane office javu padshe.",
        "Swiggy e 2 kalak pachi thandi roti deliver kari, su superfast delivery che!",
        "Food ekdum delicious hatu ane service pan fast hati, majja aavi gai."
    ]

    for sent in test_sentences:
        clean_s = clean_text(sent)
        pred = best_val_pipeline.predict([clean_s])[0]
        if hasattr(best_val_pipeline, "predict_proba"):
            probs = best_val_pipeline.predict_proba([clean_s])[0]
            sarc_idx = list(best_val_pipeline.classes_).index("Sarcasm")
            prob_sarc = probs[sarc_idx]
            prob_str = f"P(Sarcasm) = {prob_sarc * 100:.1f}%"
        else:
            prob_str = "N/A"
        
        signals = extract_linguistic_signals(sent)
        contrast = " [Contrast Detected]" if signals["sentiment_contrast"] else ""
        print(f"Input: \"{sent}\"")
        print(f"  -> Prediction: [{pred}] ({prob_str}){contrast}\n")

if __name__ == "__main__":
    main()
