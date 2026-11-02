"""
Text Preprocessor and Feature Extraction for Gujarati-English Sarcasm Detection.
Shared module for training, Colab notebook, and backend inference.
"""

import re
import string
import numpy as np

# Key Gujarati and English positive / negative sentiment cues for contrast detection
POSITIVE_CUES = {
    "wah", "mast", "saras", "perfect", "superb", "great", "zabardast", "jordar", "kamaal",
    "shandar", "relief", "delicious", "tasty", "fresh", "swadisht", "fast", "speed", "bullet",
    "rocket", "salute", "award", "nobel", "congratulations", "aabhar", "thanks", "thank",
    "satisfy", "satisfied", "happy", "khush", "khushi", "maja", "majja", "maza", "blessing",
    "genius", "einstein", "top", "distinction", "prashansaniya", "clean", "hygienic", "smooth"
}

NEGATIVE_CUES = {
    "fail", "waste", "late", "delay", "thaki", "kantali", "kharaab", "kharab", "broken",
    "damage", "damaged", "shattered", "tootyo", "crash", "stuck", "fasaai", "cancel",
    "cancelled", "problem", "takleef", "dard", "headache", "sar", "spoil", "spoiled",
    "drop", "dropped", "burst", "leak", "cut", "deduction", "strike", "disappointed",
    "sad", "stress", "tension", "cheated", "fraud", "bad", "loss", "nuksan", "potholes",
    "khadda", "traffic", "garmi", "dirt", "gandu", "mativalu", "ice", "cold", "thandu", "burnt"
}

def clean_text(text: str) -> str:
    """
    Cleans Romanized Gujarati-English text while preserving punctuation cues and word structures.
    """
    if not isinstance(text, str):
        text = str(text or "")
    
    # Normalize excessive whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Normalize elongated character repetitions (e.g., 'masttttt' -> 'mastt', 'waaah' -> 'waah')
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    return text

def extract_linguistic_signals(text: str) -> dict:
    """
    Extracts explicit linguistic features useful for sarcasm explainability and diagnostic analysis:
    - exclamation_count: count of '!'
    - question_count: count of '?'
    - has_multi_punct: presence of '!!', '?!', '??'
    - has_elongated_words: elongated spellings like 'soooo', 'waaah'
    - pos_cues_found: matched positive cues
    - neg_cues_found: matched negative cues
    - sentiment_contrast: True if both positive cues and negative cues appear together
    """
    cleaned = clean_text(text)
    lower = cleaned.lower()
    tokens = set(re.findall(r'[a-zA-Z]+', lower))
    
    exclamation_count = text.count('!')
    question_count = text.count('?')
    has_multi_punct = bool(re.search(r'[!?]{2,}', text))
    has_elongated_words = bool(re.search(r'([a-zA-Z])\1{2,}', text))
    
    pos_found = sorted(list(tokens.intersection(POSITIVE_CUES)))
    neg_found = sorted(list(tokens.intersection(NEGATIVE_CUES)))
    sentiment_contrast = (len(pos_found) > 0 and len(neg_found) > 0)
    
    return {
        "exclamation_count": exclamation_count,
        "question_count": question_count,
        "has_multi_punct": has_multi_punct,
        "has_elongated_words": has_elongated_words,
        "positive_cues": pos_found,
        "negative_cues": neg_found,
        "sentiment_contrast": sentiment_contrast,
        "char_length": len(text),
        "word_count": len(cleaned.split())
    }
