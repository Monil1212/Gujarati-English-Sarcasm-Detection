# Gujarati-English Sarcasm Detection System

This student project classifies short Romanized Gujarati-English text as **Sarcasm** or **Non-Sarcasm**. It uses a scikit-learn text pipeline, a FastAPI backend, and a small responsive HTML interface. The classifier matches patterns learned from labelled examples; it does not understand intent or context like a person.

## Technology

- Python, pandas, NumPy, scikit-learn and joblib for data and model training
- FastAPI, Pydantic and Uvicorn for the API
- Plain HTML, CSS and JavaScript for the frontend

## Dataset and preprocessing

`sarcasm_dataset.csv` contains 1,200 rows and the columns `id`, `sentence`, `label`, `source_type`, `language`, `sub_category`, and `group_id`. It has 600 examples of each label. All examples are programmatically authored or spelling variants, not collected user comments. The audit found no empty cells, exact duplicate sentences, normalized duplicate sentences, or same-text label conflicts. There are 241 groups. `understatement_irony` has only two rows; category scores for it are not reliable.

`backend/model/preprocessor.py` collapses whitespace and reduces character runs of three or more to two. It preserves punctuation. The same function is used in training and inference.

`Sarcasm_Detection_Colab.ipynb` is the supplied walkthrough for the earlier baseline. It is retained for reference; `train_model.py` is the maintained training workflow for the current model, including group-contained SVM calibration.

## Features and training

The pipeline combines word TF-IDF (1–3 word n-grams) and character-boundary TF-IDF (3–5 character n-grams). Word features capture terms and short phrases. Character features can recognize spelling variation such as `che`/`chhe`. TF-IDF downweights terms common across many examples and gives more weight to terms that are informative in a sentence.

`train_model.py` uses seed 42 and splits entire `group_id` groups into train, validation and test partitions. Logistic Regression, Complement Naive Bayes, Multinomial Naive Bayes and Random Forest are benchmarked with five stratified group folds. The SVM's `CalibratedClassifierCV` estimator contains both the TF-IDF `FeatureUnion` and `LinearSVC`; therefore, each calibration fold learns its own vocabulary and IDF values from that fold's fitting examples only. Calibration folds use `StratifiedGroupKFold` on training groups. The model with the best validation macro F1 is selected; the held-out test set is evaluated once after selection. At prediction time the saved calibrated estimator applies the same shared `clean_text` function and its fold-fitted TF-IDF pipelines.

Retraining writes the selected model and run-generated metadata to `backend/model/sarcasm_pipeline.joblib` and `backend/model/model_metadata.json`.

## Evaluation

Results are written to `backend/model/model_metadata.json`. This file is the source of truth for the selected model, scikit-learn version, settings, split sizes, calibration fold audit, test metrics, category scores and test errors. It is regenerated from the current training run; no previous-run metrics are copied into it. Results describe this authored dataset only and do not establish accuracy on real conversations.

## Run on Windows

Open PowerShell in the project folder. Python 3.10 or later is recommended.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
```

`backend/requirements.txt` pins **scikit-learn 1.9.1**, the version recorded in the delivered model metadata. To install that version directly, use `python -m pip install scikit-learn==1.9.1`. To verify your active environment, run `python -c "import sklearn; print(sklearn.__version__)"` and compare it with `scikit_learn_version` in `backend/model/model_metadata.json` before loading the joblib artifact.

Start the application:

```powershell
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. The API also provides `/api/health`, `/api/examples`, `/api/predict`, and `/api/model-info`. Model evaluation is retained in the internal metadata/API for development; there is no metrics page in the user interface.

Run the tests:

```powershell
python -m unittest discover -s backend\tests -v
```

Retrain and regenerate model artifacts:

```powershell
python train_model.py
```

## Limitations

The dataset is small and synthetic, spelling is inconsistent, sarcasm depends on context, and some categories are underrepresented. Related groups are kept together during splitting, but group quality is based on author-defined IDs. The probability is a calibrated classifier score, not a guarantee that the sentence is sarcastic. The UI therefore emphasizes the predicted class and a feature-based explanation rather than treating probability as certainty.
