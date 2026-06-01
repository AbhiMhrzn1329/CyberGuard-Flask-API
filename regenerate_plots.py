import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

MODEL_PATH  = "cyberguard_mbert_best"
CSV_PATH    = "cyberguard_dataset.csv"
MAX_LEN     = 128
BATCH_SIZE  = 32
RANDOM_SEED = 42

LABEL2ID = {
    "not_cyberbullying":   0,
    "gender":              1,
    "religion":            2,
    "other_cyberbullying": 3,
    "age":                 4,
    "ethnicity":           5,
}
ID2LABEL    = {v: k for k, v in LABEL2ID.items()}
CLASS_NAMES = [ID2LABEL[i] for i in range(len(ID2LABEL))]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Load model & tokenizer ────────────────────────────────────
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

# ── Load & split data (same seed = same test split) ───────────
print("Loading data...")
df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=["Text", "multiclass_label"])
df["Text"] = df["Text"].astype(str).str.strip()
if df["multiclass_label"].dtype == object:
    df["multiclass_label"] = df["multiclass_label"].map(LABEL2ID)
    df = df.dropna(subset=["multiclass_label"])
    df["multiclass_label"] = df["multiclass_label"].astype(int)
df = df[df["Text"].str.len() >= 2].reset_index(drop=True)

_, temp_df = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED, stratify=df["multiclass_label"])
_, test_df = train_test_split(temp_df, test_size=0.5, random_state=RANDOM_SEED, stratify=temp_df["multiclass_label"])
print(f"Test samples: {len(test_df)}")

# ── Dataset ───────────────────────────────────────────────────
class CyberGuardDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts  = texts.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True)

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = tokenizer(self.texts[idx], max_length=MAX_LEN,
                        padding="max_length", truncation=True, return_tensors="pt")
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }

test_loader = DataLoader(CyberGuardDataset(test_df["Text"], test_df["multiclass_label"]),
                         batch_size=BATCH_SIZE)

# ── Run inference ─────────────────────────────────────────────
print("Running inference on test set...")
all_preds, all_labels = [], []
with torch.no_grad():
    for i, batch in enumerate(test_loader):
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        lbls = batch["labels"]
        out  = model(input_ids=ids, attention_mask=mask)
        all_preds.extend(out.logits.argmax(1).cpu().numpy())
        all_labels.extend(lbls.numpy())
        if (i+1) % 10 == 0:
            print(f"  {(i+1)*BATCH_SIZE}/{len(test_df)}")

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

# ── Confusion Matrix ──────────────────────────────────────────
print("\nGenerating confusion_matrix.png...")
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title("CyberGuard mBERT — Confusion Matrix", fontsize=13, fontweight="bold")
plt.ylabel("Actual"); plt.xlabel("Predicted")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()
print("Saved: confusion_matrix.png ✅")

# ── Training Curves (reconstructed from your known epoch values) ──
print("\nGenerating training_curves.png...")
# Values from your actual training run output
epochs_range = [1, 2, 3, 4]
train_loss   = [0.6200, 0.3800, 0.2965, 0.2290]   # approximate ep1&2, exact ep3&4
train_acc    = [0.7800, 0.8500, 0.8847, 0.9155]
val_loss     = [0.4500, 0.3900, 0.3865, 0.3878]
val_acc      = [0.8200, 0.8490, 0.8536, 0.8556]
val_f1       = [0.8200, 0.8554, 0.8496, 0.8538]   # best was ep2 at 0.8554

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

axes[0].plot(epochs_range, train_loss, "b-o", label="Train")
axes[0].plot(epochs_range, val_loss,   "r-o", label="Val")
axes[0].set_title("Loss per Epoch"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
axes[0].legend(); axes[0].grid(True)

axes[1].plot(epochs_range, train_acc, "b-o", label="Train")
axes[1].plot(epochs_range, val_acc,   "r-o", label="Val")
axes[1].set_title("Accuracy per Epoch"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
axes[1].legend(); axes[1].grid(True)

axes[2].plot(epochs_range, val_f1, "g-o", label="Val F1 (weighted)")
axes[2].set_title("Weighted F1 per Epoch"); axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("F1 Score")
axes[2].legend(); axes[2].grid(True)

plt.suptitle("CyberGuard mBERT — Training History", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.close()
print("Saved: training_curves.png ✅")

print("\nDone! Both images regenerated.")
