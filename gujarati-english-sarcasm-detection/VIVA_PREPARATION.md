# Viva Preparation: Gujarati-English Sarcasm Detection

## Project in one minute

The project takes a Romanized Gujarati-English sentence in a browser, sends it to a FastAPI endpoint, applies the same whitespace and repeated-letter cleanup used during training, then passes the text through a saved scikit-learn pipeline. The pipeline turns the sentence into word and character TF-IDF features and predicts Sarcasm or Non-Sarcasm. The UI reports the label and describes the classifier’s learned-pattern basis without claiming to know the writer’s intent. Model evaluation is generated during training and is retained in metadata for development.

## End-to-end flow

1. `sarcasm_dataset.csv` provides labelled sentences and a `group_id` that keeps related variants together.
2. `train_model.py` cleans text and splits groups into training, validation and held-out test partitions using seed 42.
3. Training pipelines fit the word and character TF-IDF vocabularies on training data only. For the calibrated SVM, feature extraction is inside the estimator passed to `CalibratedClassifierCV`, so each calibration fold fits its own vocabulary and IDF weights.
4. Several lightweight classifiers are compared. Group-aware folds are used for cross-validation and SVM probability calibration; the calibration fold audit is stored in metadata.
5. Validation macro F1 selects the final model. Only then is its performance measured on the held-out test set.
6. The trained pipeline and actual metrics are saved to `backend/model/sarcasm_pipeline.joblib` and `backend/model/model_metadata.json`.
7. FastAPI loads those artifacts. The browser calls `/api/examples` and `/api/predict`; the server validates input, cleans it, predicts, and returns the class and a limitation-aware explanation. Calibrated probability is returned only when metadata identifies the model as calibrated.

## Important files

- `sarcasm_dataset.csv`: labelled dataset with 1,200 rows and seven columns.
- `DATASET_README.md`: dataset provenance and category information from the original project.
- `build_full_leak_free_dataset.py`: original reproducible builder for the authored dataset. Running it regenerates the data; it is not needed to run the application.
- `train_model.py`: split, features, candidate comparison, final evaluation and artifact export.
- `backend/model/preprocessor.py`: shared text cleanup and diagnostic lexical signals.
- `backend/model/sarcasm_pipeline.joblib`: trained `CalibratedClassifierCV` whose base estimator is a pipeline containing the `FeatureUnion` and `LinearSVC`.
- `backend/model/model_metadata.json`: metrics and settings from the actual training run.
- `backend/app.py`: API, request checks, model loading, examples and prediction endpoint.
- `frontend/index.html`: responsive single-page interface. There is no user-facing model metrics page.
- `backend/requirements.txt`: Python dependencies.
- `backend/tests/test_api.py`: API and dataset integrity checks.

## Environment version

The model metadata records scikit-learn **1.9.1**, and `backend/requirements.txt` pins that version for joblib compatibility. Install it with `python -m pip install scikit-learn==1.9.1` (or install the complete project dependencies with `python -m pip install -r backend\requirements.txt`). Verify the active interpreter with `python -c "import sklearn; print(sklearn.__version__)"` before loading the saved model. Do not ignore an estimator version warning; retrain under the installed version if it does not match.

## Model and features

The training script selects **Group-Calibrated Linear SVM (C=1.0)** on validation macro F1. It uses a Linear SVM because it is lightweight and works well with sparse TF-IDF features. The `CalibratedClassifierCV` estimator is a complete pipeline: `FeatureUnion` (word and character TF-IDF) followed by `LinearSVC`. Each calibration fold clones and fits that whole pipeline on its calibration-training rows only. Calibration folds are stratified and group-aware. Logistic Regression, Complement Naive Bayes, Multinomial Naive Bayes and Random Forest are benchmarked with group-aware cross-validation; all candidates are assessed on the same validation set for selection. The SVM is selected using validation macro F1; test data is not used for selection or threshold tuning. The metadata records the actual validation result and the cross-validation score is `null` for the SVM because the SVM's nested calibration is evaluated via validation rather than a separate nested outer CV.

TF-IDF represents text as numeric feature weights. A word n-gram is a sequence of adjacent words: unigram `mast`, bigram `mast service`, trigram `su mast service`. Word n-grams keep useful words and phrases. Character n-grams are adjacent character sequences, including word boundaries here. They can share partial spelling patterns across variants such as `che` and `chhe`. The union combines both feature sets into one sparse vector.

The model learns a boundary from examples paired with labels. It does not learn a complete linguistic theory of sarcasm. At inference, `app.py` applies the shared `clean_text` function and sends the resulting string to the saved calibrator; each fitted calibrated estimator then applies its saved TF-IDF transforms and Linear SVM. This is the same cleanup and feature pipeline used during calibration training. The current explanation says that the classifier learned patterns from labelled data; it does not invent emotional reasoning or attribute a human intention.

## Dataset and split

The schema is `id`, `sentence`, `label`, `source_type`, `language`, `sub_category`, and `group_id`. Labels are `Sarcasm` and `Non-Sarcasm`; each has 600 examples. All examples are programmatically authored or spelling variants. The audit found zero missing cells, exact duplicate strings, normalized duplicate strings, or normalized text with conflicting labels. The dataset contains 241 groups.

Group-aware splitting assigns whole group IDs to one partition. This reduces leakage from paraphrases or spelling variants. The split has 832 training examples across 168 groups, 171 validation examples across 36 groups and 197 test examples across 37 groups. The actual split is based on groups, so row percentages are approximate. Validation guides model selection. Test data is reserved for the final evaluation, not tuning.

## Latest held-out evaluation

Metrics below are from the current generated metadata for the selected group-calibrated SVM:

- scikit-learn version: **1.9.1**
- Validation macro F1 used to select this model: **90.64%**
- Accuracy: **83.76%** (165/197)
- Macro precision: **84.39%**
- Macro recall: **84.22%**
- Macro F1: **83.75%**
- Sarcasm precision / recall / F1: **91.01% / 77.14% / 83.51%**
- Non-Sarcasm precision / recall / F1: **77.78% / 91.30% / 84.00%**
- Confusion matrix, actual rows and predicted columns (`Non-Sarcasm`, `Sarcasm`): `[[84, 8], [24, 81]]`
- Calibration fold group overlaps: **0 in each of 5 folds**

Examples from this run’s held-out error analysis include a false positive: “Aaje hawa shital che ane sky cloudy che, garden ma walking karvani maja aave che.” It is a genuine positive sentence classified as sarcasm. A false negative is “Flight canceled, luggage lost, and no hotel provided, minor inconvenience by airline!” It is understatement irony classified as non-sarcasm. Understatement irony has only two test rows, so its category score is not dependable. These scores describe the authored dataset and do not guarantee performance on natural conversation.

## Key evaluation terms

- **Accuracy:** fraction of all test predictions that are correct.
- **Precision:** among predictions for a class, fraction that are correct.
- **Recall:** among actual examples of a class, fraction found by the model.
- **F1:** harmonic mean of precision and recall. Macro F1 calculates F1 for each class then averages them equally.
- **Confusion matrix:** counts actual versus predicted labels. Here the diagonal cells are correct predictions; off-diagonal cells are errors.
- **Calibration:** a separate mapping from classifier scores to probability estimates, fit using training-only examples. A calibrated score is still not certainty.

## Viva questions and short answers

1. **What problem does the project solve?** It classifies a Romanized Gujarati-English sentence as sarcastic or non-sarcastic.
2. **What is code-mixed text?** Text that combines words or structures from more than one language, here Gujarati and English.
3. **What does Romanized Gujarati mean?** Gujarati words written with Latin letters, often with variable spelling.
4. **Which model was selected?** A Linear SVM with C=1.0 and group-aware probability calibration.
5. **Why use TF-IDF?** It converts text to numeric features and gives relatively more weight to terms that distinguish documents.
6. **What is a unigram?** One token, such as `mast`.
7. **What is a bigram?** Two adjacent tokens, such as `mast service`.
8. **Why include character n-grams?** They can capture partial spellings and word fragments despite spelling variation.
9. **What does the SVM do?** It learns a separating boundary between the feature vectors of the two labels.
10. **What does C control?** Regularization strength trade-off: how much the fit prioritizes training examples versus a simpler margin boundary.
11. **How does training learn?** It compares feature vectors with known labels and adjusts the classifier to separate the classes.
12. **How is a new prediction produced?** Input is cleaned with `clean_text`, then passed to the saved calibrated model. Each calibrated estimator applies its fold-fitted TF-IDF pipeline and Linear SVM.
13. **How is the output label chosen?** The classifier returns the class on its learned decision side; classes are `Non-Sarcasm` and `Sarcasm`.
14. **Why split into train, validation and test?** Train fits the model, validation selects it, and test estimates final performance after selection.
15. **Why split by group ID?** Related paraphrases or spelling variants should not appear on both sides of a split.
16. **What is accuracy?** Correct predictions divided by all predictions.
17. **What is precision?** Correct predictions for a class divided by all predictions for that class.
18. **What is recall?** Correct predictions for a class divided by the actual count of that class.
19. **What is macro F1?** The average of the two class F1 scores, giving each class equal weight.
20. **What does calibration do?** It maps decision scores to probability estimates using training-only folds. Each fold refits TF-IDF and the SVM together, and group IDs do not cross the fold boundary.
21. **Why not trust confidence as certainty?** Calibration estimates frequency alignment, not truth or intent, and the dataset is small and synthetic.
22. **What is one limitation?** Romanized spelling varies, sarcasm requires context, and some categories have very few examples.
23. **Is the dataset real social-media data?** No. It is programmatically authored and includes spelling variants; it is not represented as real user comments.
24. **What is a false positive?** A genuine non-sarcastic sentence incorrectly predicted as sarcastic.
25. **What is a false negative?** A sarcastic sentence incorrectly predicted as non-sarcastic.
26. **Why is group-aware evaluation stronger than a random row split?** It better prevents the model from seeing near-duplicate examples during training and testing.
27. **What does the explanation mean?** It describes the model as learning text patterns; it does not claim to infer intent or emotion.
28. **What could improve the project next?** Collect carefully labelled, consented natural examples, audit group relationships and broaden weak categories, then evaluate on a new external set.

## Architecture to say aloud

“The dataset contains labelled Romanized Gujarati-English sentences and group IDs. I split whole groups into training, validation and test sets to reduce paraphrase leakage. The calibrated SVM contains both TF-IDF feature extraction and Linear SVM inside its estimator. That means every calibration fold builds its vocabulary and IDF values from only its training groups. I choose the model using validation macro F1, then evaluate once on the held-out test set. FastAPI applies the same text cleanup and loads the saved calibrated pipeline for predictions.”
