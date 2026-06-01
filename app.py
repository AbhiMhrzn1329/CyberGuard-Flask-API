"""
================================================
 CYBERBULLYING DETECTION — FLASK API
 (6-Class Multiclass | Multilingual mBERT)
================================================
 Labels:
   0 → Not Cyberbullying
   1 → Gender
   2 → Religion
   3 → Other Cyberbullying
   4 → Age
   5 → Ethnicity

 Languages Supported:
   English | Nepali Devanagari | Romanized Nepali | Mixed

 Endpoints:
   GET  /              → API info
   POST /predict       → Single text prediction
   POST /predict_batch → Multiple texts

 Run:
   python app.py

 Test:
   POST http://127.0.0.1:5000/predict
   Body: {"text": "tah muji ho"}
================================================
"""

import pickle
import unicodedata
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)
CORS(app)

SAVE_DIR = 'cyberguard_mbert_best'
MAX_LEN  = 128

# ── Load model on startup ──────────────────────
print("Loading model, please wait...")

tokenizer = AutoTokenizer.from_pretrained(SAVE_DIR)
model     = AutoModelForSequenceClassification.from_pretrained(SAVE_DIR)
model.eval()

try:
    with open(f'{SAVE_DIR}/label_info.pkl', 'rb') as f:
        label_info = pickle.load(f)
    ID2LABEL = label_info['id2label']
except FileNotFoundError:
    ID2LABEL = {
        0: 'Not Cyberbullying',
        1: 'Gender',
        2: 'Religion',
        3: 'Other Cyberbullying',
        4: 'Age',
        5: 'Ethnicity',
    }

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = model.to(device)

print(f"✅ Model loaded on {device}")
print(f"   Labels: {ID2LABEL}\n")

LABEL_COLORS = {
    'Not Cyberbullying'  : 'green',
    'Gender'             : 'purple',
    'Religion'           : 'orange',
    'Other Cyberbullying': 'blue',
    'Age'                : 'red',
    'Ethnicity'          : 'pink',
}

# ── Preprocessing ──────────────────────────────
def preprocess_text(text: str) -> str:
    # Normalize Devanagari unicode (fixes encoding inconsistencies)
    text = unicodedata.normalize('NFC', text)
    # Remove zero-width characters common in Devanagari copy-paste
    text = text.replace('\u200c', '').replace('\u200d', '').replace('\ufeff', '')
    return text.strip()

def is_mostly_devanagari(text: str) -> bool:
    devanagari_chars = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    return devanagari_chars > len(text) * 0.3

# ── Predict function ───────────────────────────
def predict_text(text: str):
    # Preprocess text before tokenizing
    text = preprocess_text(text)

    encoding = tokenizer(
        text,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids      = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs    = model(input_ids=input_ids, attention_mask=attention_mask)
        probs      = torch.softmax(outputs.logits, dim=1)[0]
        pred_idx   = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()

    label = ID2LABEL[pred_idx]

    all_probs = {
        ID2LABEL[i]: round(probs[i].item() * 100, 2)
        for i in range(len(ID2LABEL))
    }

    # Warn if Devanagari text with low confidence
    low_confidence_warning = is_mostly_devanagari(text)

    return {
        'label'      : label,
        'label_id'   : pred_idx,
        'confidence' : round(confidence * 100, 2),
        'is_bullying': pred_idx != 0,
        'color'      : LABEL_COLORS.get(label, 'gray'),
        'all_probs'  : all_probs,
        'warning'    : 'Low confidence — model has limited Devanagari training data' if low_confidence_warning else None
    }


# ════════════════════════════════════════════
#  ROUTE 1 — HOME
# ════════════════════════════════════════════
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "api"      : "CyberGuard Multilingual Detection API",
        "model"    : "distilbert-base-multilingual-cased (mBERT)",
        "languages": ["English", "Nepali Devanagari", "Romanized Nepali", "Mixed"],
        "labels"   : ID2LABEL,
        "endpoints": {
            "GET  /"              : "API info",
            "POST /predict"       : "Predict single text",
            "POST /predict_batch" : "Predict multiple texts"
        }
    })


# ════════════════════════════════════════════
#  ROUTE 2 — SINGLE PREDICTION
# ════════════════════════════════════════════
@app.route('/predict', methods=['POST'])
def predict_single():
    data = request.get_json()

    if not data or 'text' not in data:
        return jsonify({
            "error"  : "Missing 'text' field.",
            "example": {"text": "your comment here"}
        }), 400

    text = str(data['text']).strip()
    if not text:
        return jsonify({"error": "Text cannot be empty."}), 400

    result = predict_text(text)

    return jsonify({
        "text"       : text,
        "label"      : result['label'],
        "label_id"   : result['label_id'],
        "confidence" : result['confidence'],
        "is_bullying": result['is_bullying'],
        "color"      : result['color'],
        "all_probs"  : result['all_probs'],
        "warning"    : result['warning']
    })


# ════════════════════════════════════════════
#  ROUTE 3 — BATCH PREDICTION
# ════════════════════════════════════════════
@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    data = request.get_json()

    if not data or 'texts' not in data:
        return jsonify({
            "error"  : "Missing 'texts' field.",
            "example": {"texts": ["comment one", "comment two"]}
        }), 400

    texts = data['texts']
    if not isinstance(texts, list) or len(texts) == 0:
        return jsonify({"error": "'texts' must be a non-empty list."}), 400

    results = []
    for text in texts:
        text   = str(text).strip()
        result = predict_text(text)
        results.append({
            "text"       : text,
            "label"      : result['label'],
            "label_id"   : result['label_id'],
            "confidence" : result['confidence'],
            "is_bullying": result['is_bullying'],
            "color"      : result['color']
        })

    bullying_count = sum(1 for r in results if r['is_bullying'])

    return jsonify({
        "total"         : len(results),
        "bullying_count": bullying_count,
        "safe_count"    : len(results) - bullying_count,
        "results"       : results
    })


# ════════════════════════════════════════════
#  RUN
# ════════════════════════════════════════════
if __name__ == '__main__':
    print("🚀 Starting Flask server at http://127.0.0.1:5000\n")
    app.run(debug=False, host='0.0.0.0', port=5000)