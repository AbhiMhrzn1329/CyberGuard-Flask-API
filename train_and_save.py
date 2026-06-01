import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from collections import Counter

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)

# ── STEP 2: Configuration ────────────────────────────────────
MODEL_NAME  = "distilbert-base-multilingual-cased"
CSV_PATH    = "cyberguard_dataset.csv"
MAX_LEN     = 128
BATCH_SIZE  = 32
EPOCHS      = 6        # increased from 4
PATIENCE    = 2        # early stopping patience
LR          = 2e-5
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
print(f"Device : {device}")
if device.type == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"Model  : {MODEL_NAME}")

# ── STEP 3: Load & validate data ─────────────────────────────
df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=["Text", "multiclass_label"])
df["Text"] = df["Text"].astype(str).str.strip()

# Map string labels to integers if not already done
if df["multiclass_label"].dtype == object:
    df["multiclass_label"] = df["multiclass_label"].map(LABEL2ID)
    df = df.dropna(subset=["multiclass_label"])
    df["multiclass_label"] = df["multiclass_label"].astype(int)

# Remove extremely short texts
df = df[df["Text"].str.len() >= 2].reset_index(drop=True)

print(f"\nTotal rows after cleaning : {len(df):,}")
print("\nLanguage distribution:")
if "language" in df.columns:
    print(df["language"].value_counts().to_string())
print("\nCategory distribution:")
if "category" in df.columns:
    print(df["category"].value_counts().to_string())

# ── STEP 4: Stratified train / val / test split ───────────────
train_df, temp_df = train_test_split(
    df, test_size=0.2, random_state=RANDOM_SEED,
    stratify=df["multiclass_label"]
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, random_state=RANDOM_SEED,
    stratify=temp_df["multiclass_label"]
)

print(f"\nDataset split:")
print(f"  Train : {len(train_df):,}")
print(f"  Val   : {len(val_df):,}")
print(f"  Test  : {len(test_df):,}")

# ── STEP 5: Class weights ────────────────────────────────────
label_counts = Counter(train_df["multiclass_label"].tolist())
total = sum(label_counts.values())
class_weights = torch.tensor(
    [total / (len(LABEL2ID) * label_counts[i]) for i in range(len(LABEL2ID))],
    dtype=torch.float
).to(device)
print(f"\nClass weights: {class_weights.cpu().numpy().round(3)}")

# ── STEP 6: Tokenizer & Dataset ──────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class CyberGuardDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = texts.reset_index(drop=True)
        self.labels    = labels.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ── STEP 7: Model ────────────────────────────────────────────
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABEL2ID),
    id2label=ID2LABEL,
    label2id=LABEL2ID,
)
model.to(device)

# ── STEP 8: Loss with class weights ──────────────────────────
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

# ── STEP 9: Optimizer & scheduler ────────────────────────────
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)


# ── STEP 10: Train & eval helpers ────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss, total_correct = 0, 0
    for batch in loader:
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        lbls = batch["labels"].to(device)

        optimizer.zero_grad()
        out  = model(input_ids=ids, attention_mask=mask)
        loss = loss_fn(out.logits, lbls)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss    += loss.item()
        total_correct += (out.logits.argmax(1) == lbls).sum().item()

    return total_loss / len(loader), total_correct / len(loader.dataset)


def eval_epoch(model, loader, device):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbls = batch["labels"].to(device)

            out        = model(input_ids=ids, attention_mask=mask)
            loss       = loss_fn(out.logits, lbls)
            total_loss += loss.item()
            all_preds.extend(out.logits.argmax(1).cpu().numpy())
            all_labels.extend(lbls.cpu().numpy())

    return total_loss / len(loader), np.array(all_preds), np.array(all_labels)


def predict(text: str) -> dict:
    """
    Predict cyberbullying label for any text.
    Works with: English | Nepali Devanagari | Romanized Nepali | Code-mixed
    """
    model.eval()
    enc = tokenizer(
        text,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        logits = model(**enc).logits
    probs    = torch.softmax(logits, dim=1).cpu().numpy()[0]
    label_id = probs.argmax()
    return {
        "label":      ID2LABEL[label_id],
        "confidence": round(float(probs[label_id]), 4),
        "all_scores": {ID2LABEL[i]: round(float(p), 4) for i, p in enumerate(probs)}
    }


# ══════════════════════════════════════════════════════════════
# Everything below wrapped in if __name__ == '__main__'
# Required on Windows to prevent multiprocessing crash
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":

    train_dataset = CyberGuardDataset(
        train_df["Text"], train_df["multiclass_label"], tokenizer, MAX_LEN
    )
    val_dataset = CyberGuardDataset(
        val_df["Text"], val_df["multiclass_label"], tokenizer, MAX_LEN
    )
    test_dataset = CyberGuardDataset(
        test_df["Text"], test_df["multiclass_label"], tokenizer, MAX_LEN
    )

    # pin_memory=True for faster CPU→GPU transfer
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE,
        num_workers=0, pin_memory=True
    )

    total_steps = len(train_loader) * EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # ── STEP 11: Training loop with early stopping ────────────
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
        "val_f1":     []
    }
    best_val_f1     = 0.0
    patience_counter = 0

    print("\n" + "="*70)
    print(f"{'Epoch':<8}{'Train Loss':<14}{'Train Acc':<13}{'Val Loss':<13}{'Val Acc':<12}{'Val F1'}")
    print("="*70)

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc          = train_epoch(model, train_loader, optimizer, scheduler, device)
        vl_loss, vl_preds, vl_y = eval_epoch(model, val_loader, device)

        vl_acc = accuracy_score(vl_y, vl_preds)
        vl_f1  = f1_score(vl_y, vl_preds, average="weighted")

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)
        history["val_f1"].append(vl_f1)

        print(f"{epoch:<8}{tr_loss:<14.4f}{tr_acc:<13.4f}{vl_loss:<13.4f}{vl_acc:<12.4f}{vl_f1:.4f}", end="")

        if vl_f1 > best_val_f1:
            best_val_f1      = vl_f1
            patience_counter = 0
            model.save_pretrained("cyberguard_mbert_best")
            tokenizer.save_pretrained("cyberguard_mbert_best")
            print("  <- best saved", end="")
        else:
            patience_counter += 1
            print(f"  (no improvement {patience_counter}/{PATIENCE})", end="")
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping triggered at epoch {epoch} — best Val F1: {best_val_f1:.4f}")
                break
        print()

    print("="*70)
    print(f"\nBest Val F1 : {best_val_f1:.4f}")
    print(f"Epochs run  : {len(history['val_f1'])} / {EPOCHS}")

    # ── STEP 12: Training curves ──────────────────────────────
    epochs_range = range(1, len(history["val_f1"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(epochs_range, history["train_loss"], "b-o", label="Train")
    axes[0].plot(epochs_range, history["val_loss"],   "r-o", label="Val")
    axes[0].set_title("Loss per Epoch"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(epochs_range, history["train_acc"], "b-o", label="Train")
    axes[1].plot(epochs_range, history["val_acc"],   "r-o", label="Val")
    axes[1].set_title("Accuracy per Epoch"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].legend(); axes[1].grid(True)

    axes[2].plot(epochs_range, history["val_f1"], "g-o", label="Val F1 (weighted)")
    axes[2].set_title("Weighted F1 per Epoch"); axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("F1 Score")
    axes[2].legend(); axes[2].grid(True)

    plt.suptitle("CyberGuard mBERT — Training History", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.show()
    print("Saved: training_curves.png")

    # ── STEP 13: Final test evaluation ───────────────────────
    print("\n" + "="*70)
    print("FINAL TEST SET EVALUATION")
    print("="*70)

    _, test_preds, test_labels = eval_epoch(model, test_loader, device)

    acc  = accuracy_score(test_labels, test_preds)
    f1_w = f1_score(test_labels, test_preds, average="weighted")
    f1_m = f1_score(test_labels, test_preds, average="macro")
    prec = precision_score(test_labels, test_preds, average="weighted")
    rec  = recall_score(test_labels, test_preds, average="weighted")

    print(f"\n  Accuracy          : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision (wtd)   : {prec:.4f}")
    print(f"  Recall    (wtd)   : {rec:.4f}")
    print(f"  F1 Score  (wtd)   : {f1_w:.4f}")
    print(f"  F1 Score  (macro) : {f1_m:.4f}")
    print("\n── Per-Class Report ───────────────────────────────────────")
    print(classification_report(test_labels, test_preds, target_names=CLASS_NAMES))

    # ── STEP 14: Confusion matrix ─────────────────────────────
    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES
    )
    plt.title("CyberGuard mBERT — Confusion Matrix", fontsize=13, fontweight="bold")
    plt.ylabel("Actual"); plt.xlabel("Predicted")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()
    print("Saved: confusion_matrix.png")

    # ── STEP 15: Per-language evaluation ─────────────────────
    if "language" in df.columns:
        print("\n── Per-Language Evaluation on Test Set ───────────────────")
        test_df_eval = test_df.reset_index(drop=True).copy()
        test_df_eval["pred"] = test_preds

        for lang in ["english", "nepali_devanagari", "romanized_nepali", "nepali_mixed"]:
            subset = test_df_eval[test_df_eval["language"] == lang]
            if len(subset) == 0:
                continue
            lang_f1  = f1_score(subset["multiclass_label"], subset["pred"], average="weighted", zero_division=0)
            lang_acc = accuracy_score(subset["multiclass_label"], subset["pred"])
            print(f"  {lang:22s}  rows={len(subset):5d}  acc={lang_acc:.4f}  f1={lang_f1:.4f}")

    # ── STEP 16: Save results ─────────────────────────────────
    results_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision (weighted)", "Recall (weighted)",
                   "F1 Score (weighted)", "F1 Score (macro)"],
        "Score":  [round(acc,4), round(prec,4), round(rec,4),
                   round(f1_w,4), round(f1_m,4)]
    })
    results_df.to_csv("evaluation_results.csv", index=False)
    print("\nSaved: evaluation_results.csv")
    print(results_df.to_string(index=False))

    # ── STEP 17: Demo predictions ─────────────────────────────
    print("\n── Sample Predictions ─────────────────────────────────────")
    samples = [
        "ta randi ho",
        "tah randi ko xora ho",
        "hamro sathi randi ko ban ho",
        "तिमी त muji हिो| ",
        " तँ मुजी हो",
        "I love you mero pyaro babe",
        "you are ekdamai muji person",
        "timi ta hero rahexa",
        "you are a hero",
    ]

    print(f"\n  {'Text':<42} {'Prediction':<22} {'Confidence':<10} {'Status'}")
    print("  " + "-"*90)

    for text in samples:
        result = predict(text)
        status = "Bullying" if result["label"] != "not_cyberbullying" else "Non-Bullying"
        print(f"  {text:<40} {result['label']:<22} {result['confidence']:.2%}   {status}")