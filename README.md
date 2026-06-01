# CyberGuard Flask API

DistilBERT multilingual cyberbullying detection API.

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Place the trained model folder `cyberguard_mbert_best/` in the root directory
   (model.safetensors excluded from repo due to 516MB size — available on request)
3. Run: `python app.py`

## Endpoints
- POST /predict — single text prediction
- POST /predict_batch — batch prediction

## Labels
0: Not Cyberbullying | 1: Gender | 2: Religion | 3: Other | 4: Age | 5: Ethnicity
