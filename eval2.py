import sys, traceback
sys.stdout.flush()
print('Step 1: imports')
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch, pandas as pd
from sklearn.metrics import classification_report
print('Step 2: tokenizer')
tokenizer = AutoTokenizer.from_pretrained('cyberguard_mbert_best')
print('Step 3: model')
model = AutoModelForSequenceClassification.from_pretrained('cyberguard_mbert_best')
model.eval()
print('Step 4: csv')
df = pd.read_csv('cyberguard_dataset.csv', dtype=str, engine='python')
label2id = {'not_cyberbullying':0,'gender':1,'religion':2,'other_cyberbullying':3,'age':4,'ethnicity':5}
df['label'] = df['category'].map(label2id)
df = df.dropna(subset=['label'])
test_df = df.sample(frac=0.1, random_state=42).reset_index(drop=True)
print(f'Step 5: inference on {len(test_df)} samples')
texts = test_df['Text'].tolist()
true = test_df['label'].astype(int).tolist()
preds = []
for i in range(0, len(texts), 32):
    batch = tokenizer(texts[i:i+32], truncation=True, padding=True, max_length=128, return_tensors='pt')
    with torch.no_grad():
        out = model(**batch)
    preds.extend(out.logits.argmax(-1).tolist())
    print(f'  {len(preds)}/{len(texts)}')
    sys.stdout.flush()
report = classification_report(true, preds, target_names=list(label2id.keys()))
print(report)
open('outputs/classification_report.txt','w').write(report)
print('Saved!')
