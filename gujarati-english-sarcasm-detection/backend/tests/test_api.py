"""Route-level integration checks for the Gujarati-English sarcasm API."""
import os
import sys
import unittest

import joblib
import pandas as pd
import sklearn
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from fastapi import HTTPException
from pydantic import ValidationError

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(BACKEND_DIR, ".."))
sys.path.insert(0, BACKEND_DIR)

from app import (  # noqa: E402
    PredictionRequest,
    get_examples,
    get_model_info,
    health_check,
    model_pipeline,
    predict_sarcasm,
    clean_text,
)


class SarcasmApiTests(unittest.TestCase):
    def test_calibration_pipeline_contains_features_and_group_disjoint_folds(self):
        metadata = get_model_info()
        self.assertEqual(metadata["scikit_learn_version"], sklearn.__version__)

        pipeline = joblib.load(os.path.join(BACKEND_DIR, "model", "sarcasm_pipeline.joblib"))
        self.assertIsInstance(pipeline, CalibratedClassifierCV)
        estimator = pipeline.estimator
        self.assertIsInstance(estimator, Pipeline)
        self.assertIn("features", estimator.named_steps)
        self.assertIsInstance(estimator.named_steps["classifier"], LinearSVC)
        self.assertEqual(len(pipeline.calibrated_classifiers_), metadata["calibration"]["fold_count"])
        fold_vectorizers = []
        for calibrated_fold in pipeline.calibrated_classifiers_:
            fitted_estimator = calibrated_fold.estimator
            self.assertIsInstance(fitted_estimator, Pipeline)
            self.assertIn("features", fitted_estimator.named_steps)
            word_vectorizer = fitted_estimator.named_steps["features"].named_transformers["word_tfidf"]
            self.assertTrue(word_vectorizer.vocabulary_)
            fold_vectorizers.append(word_vectorizer)
        self.assertEqual(len({id(vectorizer) for vectorizer in fold_vectorizers}), len(fold_vectorizers))
        self.assertTrue(metadata["calibration"]["feature_extraction_inside_calibrated_estimator"])
        self.assertEqual(metadata["calibration"]["fold_group_overlap"], [0] * metadata["calibration"]["fold_count"])

        df = pd.read_csv(os.path.join(PROJECT_ROOT, "sarcasm_dataset.csv"))
        groups = df["group_id"]
        train_val_idx, _ = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42).split(
            df["sentence"], df["label"], groups=groups
        ))
        train_sub_idx, _ = next(GroupShuffleSplit(n_splits=1, test_size=0.15 / 0.85, random_state=42).split(
            df["sentence"].iloc[train_val_idx], df["label"].iloc[train_val_idx], groups=groups.iloc[train_val_idx]
        ))
        train_idx = train_val_idx[train_sub_idx]
        train_groups = groups.iloc[train_idx].reset_index(drop=True)
        self.assertEqual(len(pipeline.cv), metadata["calibration"]["fold_count"])
        for fit_idx, calibration_idx in pipeline.cv:
            self.assertFalse(set(train_groups.iloc[fit_idx]) & set(train_groups.iloc[calibration_idx]))

        sample = "Wahhhh, su mast service che!"
        cleaned = clean_text(sample)
        self.assertEqual(clean_text(cleaned), cleaned)
        self.assertEqual(
            pipeline.predict([cleaned])[0],
            predict_sarcasm(PredictionRequest(text=sample)).prediction
        )

    def test_health_and_model_loaded(self):
        self.assertEqual(health_check()["status"], "healthy")
        self.assertTrue(health_check()["model_loaded"])
        self.assertIsNotNone(model_pipeline)

    def test_model_info_matches_evaluation_metadata(self):
        info = get_model_info()
        self.assertEqual(info["test_samples"], 197)
        self.assertEqual(info["test_groups"], 37)
        self.assertEqual(info["leakage_audit"]["status"], "PASSED")
        self.assertAlmostEqual(info["metrics"]["accuracy"], sum(info["confusion_matrix"]["matrix"][i][i] for i in (0, 1)) / info["test_samples"], places=4)
        self.assertIn("Synthetic programmatically authored", info["dataset_provenance"])

    def test_examples_have_complete_fields(self):
        examples = get_examples()
        self.assertGreaterEqual(len(examples), 5)
        for item in examples:
            self.assertTrue(item["text"].strip())
            self.assertIn(item["expected"], {"Sarcasm", "Non-Sarcasm"})

    def test_diverse_predictions_are_valid_and_probabilities_sum_to_one(self):
        cases = [
            ("Wah, su mast service che!", "Sarcasm"),
            ("Exam ma fail thai ne maja aavi.", "Sarcasm"),
            ("Aaje traffic ma be kalak waste thaya, ekdum perfect.", "Sarcasm"),
            ("Food delivery 2 kalak late aavi, royal hospitality!", "Sarcasm"),
            ("Wah, kharekhar khub saras kaam karyu che, aabhar!", "Non-Sarcasm"),
            ("Aaje traffic ma be kalak waste thaya, hu thaki gayo chu.", "Non-Sarcasm"),
            ("Aaje somvar che ane office javu padshe.", "Non-Sarcasm"),
            ("Aaje report complete kari ane team e madad kari, thank you!", "Non-Sarcasm"),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                result = predict_sarcasm(PredictionRequest(text=text))
                self.assertEqual(result.prediction, expected)
                self.assertEqual(result.is_sarcastic, expected == "Sarcasm")
                self.assertIn(result.prediction, {"Sarcasm", "Non-Sarcasm"})
                if get_model_info().get("probability_calibrated"):
                    self.assertAlmostEqual(result.sarcasm_probability + result.non_sarcasm_probability, 1.0, places=5)
                    self.assertTrue(0 <= result.sarcasm_probability <= 1)
                else:
                    self.assertIsNone(result.confidence)
                self.assertTrue(result.explanation)

    def test_overlapping_positive_vocabulary_still_returns_valid_inference(self):
        # This deliberately resembles a known false positive: useful for tracking bias,
        # not an asserted label target.
        result = predict_sarcasm(PredictionRequest(text="Food delivery came early and warm, mast service!"))
        self.assertIn(result.prediction, {"Sarcasm", "Non-Sarcasm"})
        if result.confidence is not None:
            self.assertTrue(0 <= result.confidence <= 1)

    def test_input_validation(self):
        with self.assertRaises(HTTPException) as empty:
            predict_sarcasm(PredictionRequest(text="   "))
        self.assertEqual(empty.exception.status_code, 400)
        with self.assertRaises(ValidationError):
            PredictionRequest(text="test " * 300)

    def test_group_isolation_and_dataset_integrity(self):
        dataset_path = os.path.join(PROJECT_ROOT, "sarcasm_dataset.csv")
        df = pd.read_csv(dataset_path)
        self.assertEqual(df["label"].value_counts().to_dict(), {"Non-Sarcasm": 600, "Sarcasm": 600})
        self.assertEqual(df.groupby("group_id")["label"].nunique().max(), 1)
        self.assertEqual(df["sentence"].duplicated().sum(), 0)
        normalized = df["sentence"].str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()
        self.assertEqual(normalized.duplicated().sum(), 0)

        self.assertEqual(df["group_id"].nunique(), 241)
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
        train_val_idx, test_idx = next(splitter.split(df["sentence"], df["label"], groups=df["group_id"]))
        val_splitter = GroupShuffleSplit(n_splits=1, test_size=0.15 / 0.85, random_state=42)
        train_sub_idx, val_sub_idx = next(val_splitter.split(
            df["sentence"].iloc[train_val_idx], df["label"].iloc[train_val_idx],
            groups=df["group_id"].iloc[train_val_idx],
        ))
        train_idx = train_val_idx[train_sub_idx]
        val_idx = train_val_idx[val_sub_idx]
        groups = df["group_id"]
        self.assertFalse(set(groups.iloc[train_idx]) & set(groups.iloc[val_idx]))
        self.assertFalse(set(groups.iloc[train_idx]) & set(groups.iloc[test_idx]))
        self.assertFalse(set(groups.iloc[val_idx]) & set(groups.iloc[test_idx]))
        audit = get_model_info()["leakage_audit"]
        self.assertEqual(audit["train_test_overlap"], 0)
        self.assertEqual(audit["train_val_overlap"], 0)
        self.assertEqual(audit["val_test_overlap"], 0)

if __name__ == "__main__":
    unittest.main(verbosity=2)
