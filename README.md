# AI Phishing Email Detector

An AI-powered phishing email detection system built with Bidirectional LSTM deep learning, achieving **98.46% accuracy** on 82,000+ real emails.

## Live Demo
Paste any email into the web dashboard and get instant results with confidence scores and highlighted suspicious words.

## Features
- Bidirectional LSTM deep learning model
- 98.46% test accuracy on 82,486 emails
- Highlights suspicious words in red
- Real-time confidence score
- Scan history log
- Beautiful web dashboard (Flask)

## Tech Stack
| Layer | Technology |
|-------|-----------|
| AI Model | TensorFlow, Keras, Bidirectional LSTM |
| NLP | NLTK, Tokenizer, Padding |
| Backend | Python, Flask |
| Dataset | 82,486 real phishing + legitimate emails |

## Model Performance
| Metric | Score |
|--------|-------|
| Test Accuracy | 98.46% |
| Test Loss | 0.0437 |
| Training Emails | 65,980 |
| Test Emails | 16,496 |

## Installation

git clone https://github.com/shaheerulislam/AI-Phishing-Email-Detector.git
cd AI-Phishing-Email-Detector
pip install tensorflow scikit-learn pandas numpy flask nltk
\\\

Download dataset from Kaggle:
https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset
Place phishing_email.csv in the data/ folder, then:

\\\
py src/preprocess.py
py src/train.py
py dashboard/app.py
\\\

Open http://localhost:5000

## Project Structure
\\\
AI-Phishing-Email-Detector/
├── data/                  <- dataset (not included)
├── model/                 <- trained model saved here
├── src/
│   ├── preprocess.py      <- clean and tokenize emails
│   ├── train.py           <- train Bidirectional LSTM
│   └── predict.py         <- predict phishing/legitimate
├── dashboard/
│   └── app.py             <- Flask web dashboard
└── README.md
\\\

## Author
Shaheer ul Islam
