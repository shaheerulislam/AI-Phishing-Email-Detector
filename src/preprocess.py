import pandas as pd
import numpy as np
import re
import os
import pickle
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Download NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# ── Config ────────────────────────────────────────────────────────────────────
MAX_WORDS   = 10000   # vocabulary size
MAX_LEN     = 200     # max words per email
TEST_SIZE   = 0.2
RANDOM_SEED = 42

# ── Phishing keywords (used for highlighting) ─────────────────────────────────
PHISHING_KEYWORDS = [
    'verify', 'account', 'password', 'click', 'link', 'urgent', 'suspended',
    'confirm', 'update', 'bank', 'credit', 'debit', 'login', 'secure',
    'winner', 'prize', 'free', 'offer', 'limited', 'expire', 'immediately',
    'alert', 'warning', 'unusual', 'suspicious', 'blocked', 'locked',
    'validate', 'billing', 'invoice', 'payment', 'transfer', 'wire',
    'congratulations', 'selected', 'claim', 'reward', 'bonus', 'casino',
    'lottery', 'inheritance', 'million', 'dollars', 'nigerian', 'prince'
]


def clean_text(text):
    """Clean and normalize email text."""
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', ' url ', text)   # replace URLs
    text = re.sub(r'\S+@\S+', ' email ', text)         # replace emails
    text = re.sub(r'\d+', ' num ', text)               # replace numbers
    text = re.sub(r'[^a-z\s]', ' ', text)              # remove special chars
    text = re.sub(r'\s+', ' ', text).strip()           # clean whitespace

    # Remove stopwords
    stop = set(stopwords.words('english'))
    tokens = [w for w in text.split() if w not in stop and len(w) > 2]
    return ' '.join(tokens)


def load_and_clean(data_path='data/phishing_email.csv'):
    """Load dataset and clean email text."""
    print("📂 Loading dataset...")
    df = pd.read_csv(data_path)
    print(f"   ✅ Loaded {len(df):,} emails")
    print(f"   📊 Phishing : {df['label'].sum():,}")
    print(f"   📊 Legitimate: {(df['label']==0).sum():,}")

    print("\n🧹 Cleaning email text...")
    df['clean_text'] = df['text_combined'].apply(clean_text)
    df = df[df['clean_text'].str.len() > 10].reset_index(drop=True)
    print(f"   ✅ {len(df):,} emails after cleaning")
    return df


def tokenize(df, model_dir='model'):
    """Tokenize and pad email sequences."""
    os.makedirs(model_dir, exist_ok=True)

    print("\n🔤 Tokenizing text...")
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
    tokenizer.fit_on_texts(df['clean_text'])

    sequences = tokenizer.texts_to_sequences(df['clean_text'])
    X = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    y = df['label'].values

    print(f"   ✅ Vocabulary size : {len(tokenizer.word_index):,}")
    print(f"   ✅ Sequence shape  : {X.shape}")

    # Save tokenizer
    with open(os.path.join(model_dir, 'tokenizer.pkl'), 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"   💾 Tokenizer saved to '{model_dir}/'")

    return X, y, tokenizer


def split(X, y):
    """Split into train and test sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    print(f"\n📊 Train samples : {len(X_train):,}")
    print(f"   Test  samples : {len(X_test):,}")
    return X_train, X_test, y_train, y_test


def preprocess(data_path='data/phishing_email.csv', model_dir='model'):
    """Full preprocessing pipeline."""
    df               = load_and_clean(data_path)
    X, y, tokenizer  = tokenize(df, model_dir)
    X_train, X_test, y_train, y_test = split(X, y)
    print("\n✅ Preprocessing complete!")
    return X_train, X_test, y_train, y_test, tokenizer


if __name__ == '__main__':
    preprocess()