import os
import sys
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pickle
import re
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)

# ── Config ────────────────────────────────────────────────────────────────────
MAX_LEN   = 200
MODEL_DIR = 'model'

PHISHING_KEYWORDS = [
    'verify', 'account', 'password', 'click', 'link', 'urgent', 'suspended',
    'confirm', 'update', 'bank', 'credit', 'debit', 'login', 'secure',
    'winner', 'prize', 'free', 'offer', 'limited', 'expire', 'immediately',
    'alert', 'warning', 'unusual', 'suspicious', 'blocked', 'locked',
    'validate', 'billing', 'invoice', 'payment', 'transfer', 'wire',
    'congratulations', 'selected', 'claim', 'reward', 'bonus', 'casino',
    'lottery', 'inheritance', 'million', 'dollars', 'nigerian', 'prince'
]

def load_artifacts():
    model = load_model(os.path.join(MODEL_DIR, 'phishing_model.keras'))
    with open(os.path.join(MODEL_DIR, 'tokenizer.pkl'), 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

def clean_text(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', ' url ', text)
    text = re.sub(r'\S+@\S+', ' email ', text)
    text = re.sub(r'\d+', ' num ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    stop = set(stopwords.words('english'))
    tokens = [w for w in text.split() if w not in stop and len(w) > 2]
    return ' '.join(tokens)

def get_suspicious_words(text):
    words = text.lower().split()
    found = [w for w in words if w in PHISHING_KEYWORDS]
    return list(set(found))

def predict_email(text, model, tokenizer):
    cleaned = clean_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')
    
    prob = float(model.predict(padded, verbose=0)[0][0])
    is_phishing = prob >= 0.5
    confidence = prob if is_phishing else 1 - prob
    suspicious_words = get_suspicious_words(text)

    return {
        'is_phishing'      : is_phishing,
        'confidence'       : round(confidence * 100, 2),
        'probability'      : round(prob * 100, 2),
        'suspicious_words' : suspicious_words,
        'verdict'          : '🚨 PHISHING' if is_phishing else '✅ LEGITIMATE'
    }

if __name__ == '__main__':
    print("🔍 Loading model...", flush=True)
    model, tokenizer = load_artifacts()
    print("✅ Model loaded!\n", flush=True)

    # Test emails
    test_emails = [
        """Your account has been suspended! Click here immediately to verify 
        your password and banking details or your account will be permanently blocked.""",

        """Hi team, please find attached the meeting notes from yesterday's 
        project sync. Let me know if you have any questions."""
    ]

    for i, email in enumerate(test_emails, 1):
        print(f"── Email {i} {'─'*50}", flush=True)
        print(f"📧 Text: {email[:80]}...\n", flush=True)
        result = predict_email(email, model, tokenizer)
        print(f"   Verdict     : {result['verdict']}", flush=True)
        print(f"   Confidence  : {result['confidence']}%", flush=True)
        print(f"   Probability : {result['probability']}%", flush=True)
        print(f"   Suspicious  : {result['suspicious_words']}\n", flush=True)