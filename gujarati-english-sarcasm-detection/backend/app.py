"""
FastAPI Backend Server for Gujarati-English (Romanized) Sarcasm Detection.
Loads the trained ML pipeline, validates inputs, extracts linguistic cues,
and serves real-time inference and authentic evaluation metrics.
"""

import os
import sys
import json
import joblib
from datetime import datetime
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Ensure model directory is importable
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
MODEL_DIR = os.path.join(CURRENT_DIR, "model")
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")
sys.path.insert(0, MODEL_DIR)

from preprocessor import clean_text, extract_linguistic_signals

# Paths
MODEL_PATH = os.path.join(MODEL_DIR, "sarcasm_pipeline.joblib")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")

# Initialize FastAPI App
app = FastAPI(
    title="Gujarati-English Sarcasm Detection API",
    description="Real-time binary sarcasm classifier for Romanized Gujarati, Gujarati-English code-mixed, and English text.",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and metadata variables
model_pipeline = None
model_metadata = {}

def load_artifacts():
    global model_pipeline, model_metadata
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model artifact not found at: {MODEL_PATH}")
    
    print(f"Loading trained ML model pipeline from: {MODEL_PATH}")
    model_pipeline = joblib.load(MODEL_PATH)
    print("Model pipeline successfully loaded.")

    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            model_metadata = json.load(f)
        print("Model evaluation metadata successfully loaded.")

# Load immediately on module load
try:
    load_artifacts()
except Exception as e:
    print(f"Warning on immediate artifact load: {e}")

# Also retain startup event
@app.on_event("startup")
def startup_event():
    if model_pipeline is None:
        load_artifacts()

# Pydantic Schemas
class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000, description="Input text to classify for sarcasm.")

class LinguisticSignals(BaseModel):
    exclamation_count: int
    question_count: int
    has_multi_punct: bool
    has_elongated_words: bool
    positive_cues: List[str]
    negative_cues: List[str]
    sentiment_contrast: bool
    char_length: int
    word_count: int

class PredictionResponse(BaseModel):
    text: str
    cleaned_text: str
    prediction: str
    confidence: Optional[float] = None
    sarcasm_probability: Optional[float] = None
    non_sarcasm_probability: Optional[float] = None
    is_sarcastic: bool
    explanation: str
    linguistic_signals: LinguisticSignals
    model_name: str
    timestamp: str

# Endpoints
@app.get("/")
def serve_frontend():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Gujarati-English Sarcasm Detection API is active. Frontend index.html not found."}

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model_pipeline is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/model-info")
def get_model_info():
    if not model_metadata:
        raise HTTPException(status_code=500, detail="Model metadata is not loaded.")
    return model_metadata

@app.get("/api/examples")
def get_examples():
    return [
        {
            "category": "Mock Praise (Sarcasm)",
            "text": "Wah, su mast service che, 3 kalak pachi thandi chai aapi!",
            "expected": "Sarcasm"
        },
        {
            "category": "Sentiment Contrast (Sarcasm)",
            "text": "Exam ma fail thai ne maja aavi.",
            "expected": "Sarcasm"
        },
        {
            "category": "Traffic & Irony (Sarcasm)",
            "text": "Aaje traffic ma be kalak waste thaya, ekdum perfect.",
            "expected": "Sarcasm"
        },
        {
            "category": "Rhetorical Ridicule (Sarcasm)",
            "text": "Tari vaat sambhali ne toh duniya badlai gai, Nobel prize aapo aane koi.",
            "expected": "Sarcasm"
        },
        {
            "category": "E-Commerce Irony (Sarcasm)",
            "text": "Superb yaar, 25 divas pachi shoes deliver thayu, ekdum bullet train speed!",
            "expected": "Sarcasm"
        },
        {
            "category": "Genuine Praise (Non-Sarcastic)",
            "text": "Wah, kharekhar khub saras kaam karyu che, aabhar!",
            "expected": "Non-Sarcasm"
        },
        {
            "category": "Genuine Relief (Non-Sarcastic)",
            "text": "Finally kaam puru thayu, what a relief!",
            "expected": "Non-Sarcasm"
        },
        {
            "category": "Genuine Complaint (Non-Sarcastic)",
            "text": "Aaje traffic ma be kalak waste thaya, hu ekdum thaki gayo chu.",
            "expected": "Non-Sarcasm"
        },
        {
            "category": "Genuine Feedback (Non-Sarcastic)",
            "text": "Food ekdum delicious hatu ane service pan fast hati, majja aavi gai.",
            "expected": "Non-Sarcasm"
        },
        {
            "category": "Factual Observation (Non-Sarcastic)",
            "text": "Aaje somvar che ane office javu padshe.",
            "expected": "Non-Sarcasm"
        }
    ]

@app.post("/api/predict", response_model=PredictionResponse)
def predict_sarcasm(request: PredictionRequest):
    if not model_pipeline:
        raise HTTPException(status_code=500, detail="Model pipeline is not loaded.")

    raw_text = request.text.strip()
    if not raw_text:
        raise HTTPException(status_code=400, detail="Input sentence cannot be empty or solely whitespace.")

    cleaned = clean_text(raw_text)
    
    # Run Inference
    try:
        pred_label = model_pipeline.predict([cleaned])[0]
        
        # Only expose probabilities when the training metadata confirms that
        # the selected model was calibrated.
        classes = list(model_pipeline.classes_)
        sarc_idx = classes.index("Sarcasm")
        nonsarc_idx = classes.index("Non-Sarcasm")
        confidence = sarc_prob = nonsarc_prob = None
        if model_metadata.get("probability_calibrated") and hasattr(model_pipeline, "predict_proba"):
            probabilities = model_pipeline.predict_proba([cleaned])[0]
            sarc_prob = float(probabilities[sarc_idx])
            nonsarc_prob = float(probabilities[nonsarc_idx])
            confidence = sarc_prob if pred_label == "Sarcasm" else nonsarc_prob

        # Extract diagnostic linguistic signals
        signals = extract_linguistic_signals(raw_text)

        is_sarcastic = (pred_label == "Sarcasm")
        explanation = f"The trained text classifier predicts {pred_label}. Its decision is based on patterns learned from labelled examples; it does not determine the writer's intent."
        return PredictionResponse(
            text=raw_text,
            cleaned_text=cleaned,
            prediction=pred_label,
            confidence=round(confidence, 4) if confidence is not None else None,
            sarcasm_probability=round(sarc_prob, 4) if sarc_prob is not None else None,
            non_sarcasm_probability=round(nonsarc_prob, 4) if nonsarc_prob is not None else None,
            is_sarcastic=is_sarcastic,
            explanation=explanation,
            linguistic_signals=LinguisticSignals(**signals),
            model_name=model_metadata.get("model_name", "Trained ML Classifier"),
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        print(f"Prediction failed: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail="Prediction could not be completed. Please try again.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
