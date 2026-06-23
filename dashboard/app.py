import os
import sys
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from flask import Flask, render_template_string, request, jsonify
from predict import load_artifacts, predict_email
import json
from datetime import datetime

app = Flask(__name__)

# Load model once at startup
print("🔍 Loading model...", flush=True)
model, tokenizer = load_artifacts()
print("✅ Model ready!", flush=True)

# In-memory history log
history = []

HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Phishing Detector</title>
<style>
  :root {
    --bg: #0a0e1a;
    --card: #111827;
    --border: #1f2937;
    --accent: #6366f1;
    --danger: #ef4444;
    --success: #10b981;
    --text: #f9fafb;
    --muted: #6b7280;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'Segoe UI', sans-serif;
    min-height: 100vh;
  }

  /* ── Header ── */
  header {
    background: linear-gradient(135deg, #1e1b4b 0%, #0a0e1a 100%);
    border-bottom: 1px solid var(--border);
    padding: 1.5rem 2rem;
    display: flex;
    align-items: center;
    gap: 1rem;
  }
  header .logo { font-size: 2rem; }
  header h1 { font-size: 1.5rem; font-weight: 700; }
  header p  { color: var(--muted); font-size: 0.85rem; }
  .badge {
    margin-left: auto;
    background: rgba(99,102,241,0.15);
    border: 1px solid var(--accent);
    color: var(--accent);
    padding: 0.3rem 0.8rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
  }

  /* ── Layout ── */
  main {
    max-width: 1100px;
    margin: 2rem auto;
    padding: 0 1.5rem;
    display: grid;
    grid-template-columns: 1fr 380px;
    gap: 1.5rem;
  }

  /* ── Card ── */
  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
  }
  .card h2 {
    font-size: 1rem;
    font-weight: 600;
    margin-bottom: 1rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  /* ── Textarea ── */
  textarea {
    width: 100%;
    height: 220px;
    background: #0d1117;
    border: 1px solid var(--border);
    border-radius: 10px;
    color: var(--text);
    font-size: 0.95rem;
    padding: 1rem;
    resize: vertical;
    outline: none;
    transition: border-color 0.2s;
    font-family: inherit;
  }
  textarea:focus { border-color: var(--accent); }

  /* ── Button ── */
  button {
    margin-top: 1rem;
    width: 100%;
    padding: 0.85rem;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    border: none;
    border-radius: 10px;
    color: #fff;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
    transition: opacity 0.2s, transform 0.1s;
  }
  button:hover   { opacity: 0.9; }
  button:active  { transform: scale(0.98); }
  button:disabled { opacity: 0.5; cursor: not-allowed; }

  /* ── Result box ── */
  #result { margin-top: 1.5rem; display: none; }
  .verdict-box {
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1rem;
  }
  .verdict-box.phishing  { background: rgba(239,68,68,0.1);  border: 1px solid var(--danger); }
  .verdict-box.legit     { background: rgba(16,185,129,0.1); border: 1px solid var(--success); }
  .verdict-icon  { font-size: 2.5rem; }
  .verdict-label { font-size: 1.4rem; font-weight: 700; }
  .verdict-conf  { font-size: 0.85rem; color: var(--muted); margin-top: 0.2rem; }

  /* ── Confidence bar ── */
  .bar-wrap {
    background: #1f2937;
    border-radius: 999px;
    height: 8px;
    overflow: hidden;
    margin: 0.4rem 0 1rem;
  }
  .bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.6s ease;
  }
  .bar-fill.phishing { background: linear-gradient(90deg, #ef4444, #f97316); }
  .bar-fill.legit    { background: linear-gradient(90deg, #10b981, #34d399); }
  .bar-label { font-size: 0.8rem; color: var(--muted); }

  /* ── Suspicious words ── */
  .tags { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.5rem; }
  .tag {
    background: rgba(239,68,68,0.15);
    border: 1px solid rgba(239,68,68,0.3);
    color: #fca5a5;
    padding: 0.2rem 0.6rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 500;
  }
  .no-tags { color: var(--muted); font-size: 0.85rem; }

  /* ── History ── */
  .history-list { max-height: 520px; overflow-y: auto; }
  .history-item {
    padding: 0.75rem;
    border-radius: 10px;
    border: 1px solid var(--border);
    margin-bottom: 0.6rem;
    cursor: pointer;
    transition: border-color 0.2s;
  }
  .history-item:hover { border-color: var(--accent); }
  .history-item.phishing { border-left: 3px solid var(--danger); }
  .history-item.legit    { border-left: 3px solid var(--success); }
  .history-top { display: flex; justify-content: space-between; align-items: center; }
  .history-verdict { font-size: 0.8rem; font-weight: 600; }
  .history-verdict.phishing { color: var(--danger); }
  .history-verdict.legit    { color: var(--success); }
  .history-time  { font-size: 0.7rem; color: var(--muted); }
  .history-preview { font-size: 0.78rem; color: var(--muted); margin-top: 0.3rem;
                     white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .empty-history { color: var(--muted); font-size: 0.85rem; text-align: center; padding: 2rem 0; }

  /* ── Stats row ── */
  .stats {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.75rem;
    margin-bottom: 1.5rem;
  }
  .stat-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
  }
  .stat-num  { font-size: 1.8rem; font-weight: 700; }
  .stat-num.danger  { color: var(--danger); }
  .stat-num.success { color: var(--success); }
  .stat-num.accent  { color: var(--accent); }
  .stat-label { font-size: 0.75rem; color: var(--muted); margin-top: 0.2rem; }

  /* ── Spinner ── */
  .spinner {
    display: inline-block;
    width: 18px; height: 18px;
    border: 2px solid rgba(255,255,255,0.3);
    border-top-color: #fff;
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
    vertical-align: middle;
    margin-right: 0.5rem;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* ── Highlighted text ── */
  #highlighted-text {
    background: #0d1117;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem;
    font-size: 0.88rem;
    line-height: 1.7;
    margin-top: 0.5rem;
    max-height: 150px;
    overflow-y: auto;
    word-break: break-word;
  }
  mark {
    background: rgba(239,68,68,0.3);
    color: #fca5a5;
    border-radius: 3px;
    padding: 0 2px;
  }

  @media (max-width: 768px) {
    main { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>

<header>
  <div class="logo">🛡️</div>
  <div>
    <h1>AI Phishing Detector</h1>
    <p>Powered by Bidirectional LSTM · 98.46% Accuracy</p>
  </div>
  <div class="badge">⚡ Live</div>
</header>

<main>
  <!-- Left column -->
  <div>
    <!-- Stats -->
    <div class="stats">
      <div class="stat-card">
        <div class="stat-num accent" id="stat-total">0</div>
        <div class="stat-label">Total Scanned</div>
      </div>
      <div class="stat-card">
        <div class="stat-num danger" id="stat-phishing">0</div>
        <div class="stat-label">Phishing Found</div>
      </div>
      <div class="stat-card">
        <div class="stat-num success" id="stat-legit">0</div>
        <div class="stat-label">Legitimate</div>
      </div>
    </div>

    <!-- Input -->
    <div class="card">
      <h2>📧 Paste Email Content</h2>
      <textarea id="email-input" placeholder="Paste the full email text here..."></textarea>
      <button id="scan-btn" onclick="scanEmail()">🔍 Scan Email</button>

      <!-- Result -->
      <div id="result">
        <div class="verdict-box" id="verdict-box">
          <div class="verdict-icon" id="verdict-icon"></div>
          <div>
            <div class="verdict-label" id="verdict-label"></div>
            <div class="verdict-conf"  id="verdict-conf"></div>
          </div>
        </div>

        <div class="bar-label">Phishing Probability</div>
        <div class="bar-wrap">
          <div class="bar-fill" id="bar-fill" style="width:0%"></div>
        </div>

        <div class="card" style="margin-top:0.5rem; padding:1rem;">
          <h2>🚩 Suspicious Words</h2>
          <div id="suspicious-words"></div>
        </div>

        <div class="card" style="margin-top:0.75rem; padding:1rem;">
          <h2>🔦 Highlighted Email</h2>
          <div id="highlighted-text"></div>
        </div>
      </div>
    </div>
  </div>

  <!-- Right column: History -->
  <div class="card">
    <h2>🕒 Scan History</h2>
    <div class="history-list" id="history-list">
      <div class="empty-history">No scans yet.<br>Paste an email to get started.</div>
    </div>
  </div>
</main>

<script>
  let stats = { total: 0, phishing: 0, legit: 0 };
  let historyData = [];

  async function scanEmail() {
    const text = document.getElementById('email-input').value.trim();
    if (!text) { alert('Please paste an email first!'); return; }

    const btn = document.getElementById('scan-btn');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Scanning...';

    try {
      const res  = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: text })
      });
      const data = await res.json();
      showResult(data, text);
      addHistory(data, text);
      updateStats(data.is_phishing);
    } catch(e) {
      alert('Error: ' + e.message);
    } finally {
      btn.disabled = false;
      btn.innerHTML = '🔍 Scan Email';
    }
  }

  function showResult(data, rawText) {
    const isPhishing = data.is_phishing;
    document.getElementById('result').style.display = 'block';

    // Verdict box
    const box = document.getElementById('verdict-box');
    box.className = 'verdict-box ' + (isPhishing ? 'phishing' : 'legit');
    document.getElementById('verdict-icon').textContent  = isPhishing ? '🚨' : '✅';
    document.getElementById('verdict-label').textContent = isPhishing ? 'PHISHING DETECTED' : 'LEGITIMATE EMAIL';
    document.getElementById('verdict-conf').textContent  =
      `Confidence: ${data.confidence}%  ·  Probability: ${data.probability}%`;

    // Bar
    const bar = document.getElementById('bar-fill');
    bar.className = 'bar-fill ' + (isPhishing ? 'phishing' : 'legit');
    bar.style.width = data.probability + '%';

    // Suspicious words
    const sw = document.getElementById('suspicious-words');
    if (data.suspicious_words.length) {
      sw.innerHTML = data.suspicious_words.map(w =>
        `<span class="tag">⚠️ ${w}</span>`).join('');
      sw.className = 'tags';
    } else {
      sw.innerHTML = '<span class="no-tags">No suspicious keywords found.</span>';
      sw.className = '';
    }

    // Highlighted text
    let highlighted = rawText.replace(/</g,'&lt;').replace(/>/g,'&gt;');
    data.suspicious_words.forEach(w => {
      const re = new RegExp('\\b' + w + '\\b', 'gi');
      highlighted = highlighted.replace(re, m => `<mark>${m}</mark>`);
    });
    document.getElementById('highlighted-text').innerHTML = highlighted;
  }

  function addHistory(data, text) {
    const now = new Date().toLocaleTimeString();
    historyData.unshift({ data, text, time: now });

    const list = document.getElementById('history-list');
    const item = document.createElement('div');
    const cls  = data.is_phishing ? 'phishing' : 'legit';
    item.className = `history-item ${cls}`;
    item.innerHTML = `
      <div class="history-top">
        <span class="history-verdict ${cls}">${data.verdict}</span>
        <span class="history-time">${now}</span>
      </div>
      <div class="history-preview">${text.substring(0,80)}...</div>
      <div style="font-size:0.75rem;color:var(--muted);margin-top:0.3rem">
        Confidence: ${data.confidence}%
      </div>`;
    item.onclick = () => {
      document.getElementById('email-input').value = text;
      showResult(data, text);
    };

    if (list.querySelector('.empty-history')) list.innerHTML = '';
    list.prepend(item);
  }

  function updateStats(isPhishing) {
    stats.total++;
    isPhishing ? stats.phishing++ : stats.legit++;
    document.getElementById('stat-total').textContent    = stats.total;
    document.getElementById('stat-phishing').textContent = stats.phishing;
    document.getElementById('stat-legit').textContent    = stats.legit;
  }

  // Allow Ctrl+Enter to scan
  document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('email-input').addEventListener('keydown', e => {
      if (e.ctrlKey && e.key === 'Enter') scanEmail();
    });
  });
</script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/predict', methods=['POST'])
def predict():
    data  = request.get_json()
    email = data.get('email', '')
    result = predict_email(email, model, tokenizer)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=False, port=5000)