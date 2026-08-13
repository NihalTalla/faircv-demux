"""
Transformer tier for the blind-bio gender-leakage bound
========================================================
Separate step (CPU encoding of 24k bios is slow): encodes the blind
bios with the cached all-MiniLM-L6-v2 sentence transformer (fallback:
distilbert-base-uncased), fits the same LR head used by the other tiers,
appends the result to results/bio_leak_strong.csv and regenerates
results/fig8_leak_ladder.png with all four tiers.

Same protocol as bio_leak_strong.py: fit on train (19,200), evaluate on
test (4,800), seed 42. Frozen (feature extraction), not fine-tuned.
"""

import os
import re
import sys
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

RNG_SEED = 42
RESULTS_DIR = "results"
MAX_LEN = int(sys.argv[1]) if len(sys.argv) > 1 else 160   # token cap (re-run at 256 to check)
BATCH = 96
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ── Load & clean (identical to bio_arm.py) ───────────────────────────────────
db = np.load("FairCVdb.npy", allow_pickle=True).item()
P_tr, P_te = db["Profiles Train"], db["Profiles Test"]
B_tr, B_te = db["Bios Train"], db["Bios Test"]
g_tr, g_te = P_tr[:, 1].astype(int), P_te[:, 1].astype(int)


def clean_text(txt):
    t = txt.replace("_", " ")
    t = re.sub(r"\s+'s\b", "", t)
    return re.sub(r"\s+", " ", t).strip()


bio_tr = [clean_text(t) for t in B_tr[:, 1]]
bio_te = [clean_text(t) for t in B_te[:, 1]]
print("=" * 74)
print(f"Transformer tier: {MODEL_NAME} (frozen) + LR — blind-bio gender leak")
print("=" * 74)
print(f"  train {len(bio_tr)} | test {len(bio_te)}")

import torch
torch.set_num_threads(12)
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score

# Prefer MiniLM; fall back to distilbert if it cannot be loaded offline.
try:
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    enc_model = AutoModel.from_pretrained(MODEL_NAME)
    used = MODEL_NAME
except Exception as e:
    print(f"  MiniLM unavailable ({type(e).__name__}), falling back to distilbert")
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    enc_model = AutoModel.from_pretrained("distilbert-base-uncased")
    used = "distilbert-base-uncased"
enc_model.eval()
print(f"  encoder: {used}")


def embed(texts, bs=BATCH):
    outs = []
    t_start = time.time()
    for s in range(0, len(texts), bs):
        chunk = texts[s:s + bs]
        enc = tok(chunk, truncation=True, max_length=MAX_LEN, padding=True,
                  return_tensors="pt")
        with torch.no_grad():
            out = enc_model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            vec = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        outs.append(vec.numpy())
        if (s // bs) % 40 == 0:
            print(f"    encoded {min(s + bs, len(texts))}/{len(texts)} "
                  f"({time.time() - t_start:.0f}s)")
    return np.concatenate(outs, axis=0)


print(f"\n  encoding train ({len(bio_tr)}) …")
E_tr = embed(bio_tr)
print(f"  encoding test ({len(bio_te)}) …")
E_te = embed(bio_te)
print(f"  embeddings train {E_tr.shape} test {E_te.shape}")

pipe = Pipeline([("scaler", StandardScaler()),
                 ("clf", LogisticRegression(max_iter=1000, C=1.0,
                                            random_state=RNG_SEED))])
pipe.fit(E_tr, g_tr)
p = pipe.predict_proba(E_te)[:, 1]
acc = accuracy_score(g_te, pipe.predict(E_te))
auc = roc_auc_score(g_te, p)
print(f"  T3  {used} (frozen) + LR   acc={acc:.4f}  AUC={auc:.4f}")

# ── Append + regenerate combined figure ──────────────────────────────────────
csv_path = os.path.join(RESULTS_DIR, "bio_leak_strong.csv")
if not os.path.exists(csv_path):
    raise SystemExit("results/bio_leak_strong.csv not found — run bio_leak_strong.py (T0-T2) first.")
df = pd.read_csv(csv_path)
df = df[df["tier"] != "T3"]
df = pd.concat([df, pd.DataFrame([{"tier": "T3", "features": f"{used} (frozen, @{MAX_LEN}tok) + LR",
                                   "accuracy": acc, "auc": auc, "train_auc": np.nan}])],
               ignore_index=True)
df.to_csv(csv_path, index=False)
print(f"  appended T3 to {csv_path}")

fig, ax = plt.subplots(figsize=(9, 4.5))
order = df.sort_values("auc").iloc[::-1]
bar_colors = {"T0": "#8c8c8c", "T1": "#4878CF", "T2": "#D65F5F", "T3": "#9B59B6"}
bars = ax.barh(np.arange(len(order)), order["auc"],
               color=[bar_colors[t] for t in order["tier"]], alpha=0.85)
ax.set_yticks(np.arange(len(order)))
ax.set_yticklabels([f"{r.tier}  ({r.features})" for r in order.itertuples()], fontsize=9)
ax.set_xlabel("Test AUC (gender from blind bios)")
ax.set_title("Blind-bio gender leak — model ladder (no stronger model exceeded TF-IDF)", fontsize=11)
for b, a in zip(bars, order["auc"]):
    ax.text(a + 0.004, b.get_y() + b.get_height() / 2, f"{a:.4f}", va="center", fontsize=9)
ax.axvline(0.5, color="gray", lw=0.8, ls="--", label="chance")
ax.set_xlim(0, 1.0)
ax.legend(fontsize=8)
plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "fig8_leak_ladder.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  saved results/fig8_leak_ladder.png")
print("\n  Final leakage ladder (blind bios -> gender, test AUC):")
for r in df.sort_values("auc").itertuples():
    print(f"    {r.tier:<10s} {r.features:<40s} {r.auc:.4f}")
print("\nTransformer tier complete.")
