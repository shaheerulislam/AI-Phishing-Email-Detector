import numpy as np
import pickle
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # silence the TF warning

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ── Config ────────────────────────────────────────────────────────────────────
MAX_WORDS  = 10000
MAX_LEN    = 200
EMBED_DIM  = 64
EPOCHS     = 10
BATCH_SIZE = 128
MODEL_DIR  = 'model'

def load_data():
    print("📂 Loading preprocessed data...", flush=True)
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from preprocess import preprocess
    X_train, X_test, y_train, y_test, _ = preprocess()
    return X_train, X_test, y_train, y_test

def build_model():
    print("\n🏗️  Building LSTM model...", flush=True)
    model = Sequential([
        Embedding(MAX_WORDS, EMBED_DIM, input_length=MAX_LEN),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(32)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

def train():
    X_train, X_test, y_train, y_test = load_data()
    model = build_model()

    print("\n🚀 Training model...", flush=True)
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ModelCheckpoint(os.path.join(MODEL_DIR, 'phishing_model.keras'),
                        save_best_only=True, monitor='val_loss')
    ]

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=callbacks
    )

    print("\n📊 Evaluating...", flush=True)
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"   ✅ Test Accuracy : {acc*100:.2f}%")
    print(f"   ✅ Test Loss     : {loss:.4f}")
    print(f"\n💾 Model saved to '{MODEL_DIR}/phishing_model.keras'")
    print("\n✅ Training complete!", flush=True)

if __name__ == '__main__':
    train()