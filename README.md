# AI Phishing Email Detector

An AI-powered phishing email detection system built with Bidirectional LSTM deep learning, achieving **98.46% accuracy** on 82,000+ real emails.

## Features
- Bidirectional LSTM deep learning model
- 98.46% test accuracy on 82,486 emails
- Highlights suspicious words in red
- Real-time confidence score with probability bar
- Scan history log
- Web dashboard built with Flask

## How to Run

**1. Install dependencies**
pip install tensorflow scikit-learn pandas numpy flask nltk

**2. Download the dataset**

Download from Kaggle: https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset

Place ``phishing_email.csv`` inside the ``data/`` folder.

**3. Preprocess the data**
py src/preprocess.py

**4. Train the model**
py src/train.py

**5. Launch the dashboard**
py dashboard/app.py

**6. Open in browser**
http://localhost:5000


## Results
| Metric | Value |
|--------|-------|
| Accuracy | 98.46% |
| Test Loss | 0.0437 |
| Training samples | 65,980 |
| Test samples | 16,496 |
| Total dataset | 82,486 emails |

## Tech Stack
| Layer | Technology |
|-------|------------|
| AI Model | TensorFlow, Keras, Bidirectional LSTM |
| NLP | NLTK, Tokenizer, Sequence Padding |
| Backend | Python 3.9, Flask |

## Project Structure

AI-Phishing-Email-Detector/
├── data/               <- dataset (not included, see step 3)
├── model/              <- trained model saved here
├── src/
│   ├── preprocess.py   <- cleans and tokenizes emails
│   ├── train.py        <- trains the Bidirectional LSTM
│   └── predict.py      <- predicts phishing or legitimate
├── dashboard/
│   └── app.py          <- Flask web dashboard
└── README.md


## Author
**Shaheer ul Islam**
