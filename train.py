"""train.py - train 1D-CNN / BiLSTM / Transformer on MIT-BIH and write all artifacts.

Inter-patient protocol: train on DS1 records, test on DS2 records (no patient overlap).

    python train.py                 # full run (GPU recommended, CPU works)
    python train.py --quick         # tiny run for a fast sanity check / in-app demo
    python train.py --models cnn    # only one model
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import ecg_pipeline as ep
from models import MODEL_REGISTRY, predict_proba

K = len(ep.CLASSES)


def train_model(key, Xtr, Rtr, ytr, Xva, Rva, yva, epochs, device, out_path, seed):
    torch.manual_seed(seed)
    model = MODEL_REGISTRY[key]().to(device)
    counts = np.bincount(ytr, minlength=K).astype(float)
    w = 1.0 / np.sqrt(np.maximum(counts, 1))
    w = w / w.sum() * K
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device))
    dl = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(Rtr), torch.tensor(ytr)),
                    batch_size=256, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=epochs * len(dl))
    best, history = -1.0, []
    for epoch in range(epochs):
        model.train()
        tot = 0.0
        for xb, rb, yb in dl:
            xb, rb, yb = xb.to(device), rb.to(device), yb.to(device)
            xb = xb + 0.02 * torch.randn_like(xb)           # light noise augmentation
            loss = crit(model(xb, rb), yb)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(xb)
        p = predict_proba(model, Xva, Rva, device)
        f1 = f1_score(yva, p.argmax(1), average="macro", labels=list(range(K)), zero_division=0)
        history.append(dict(epoch=epoch + 1, train_loss=tot / len(Xtr), val_macro_f1=float(f1)))
        print(f"  [{key}] epoch {epoch + 1}/{epochs}  loss {tot / len(Xtr):.4f}  val macro-F1 {f1:.4f}")
        if f1 > best:
            best = f1
            torch.save(model.state_dict(), out_path)
    model.load_state_dict(torch.load(out_path, map_location=device, weights_only=True))
    return model, history


def evaluate(y, prob):
    pred = prob.argmax(1)
    labels = list(range(K))
    prec, rec, f1, sup = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y, pred, labels=labels)
    total = cm.sum()
    per = {}
    for i, c in enumerate(ep.CLASSES):
        tp = cm[i, i]; fn = cm[i].sum() - tp; fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        per[c] = dict(precision=float(prec[i]), recall=float(rec[i]), f1=float(f1[i]),
                      specificity=float(tn / (tn + fp)) if (tn + fp) else 0.0,
                      support=int(sup[i]))
    present = [i for i in labels if sup[i] > 0]
    return dict(accuracy=float((pred == y).mean()),
                macro_f1=float(np.mean(f1[present])),
                weighted_f1=float(np.average(f1, weights=np.maximum(sup, 0)) if sup.sum() else 0),
                per_class=per, confusion_matrix=cm.tolist())


def robustness(model, X, R, y, device, rng):
    """Macro-F1 / accuracy when Gaussian noise is added at a given SNR (dB)."""
    out = {}
    power = (X ** 2).mean(1, keepdims=True)
    for snr in [None, 20, 10, 5]:
        Xn = X if snr is None else X + rng.normal(size=X.shape).astype(np.float32) * np.sqrt(
            power / (10 ** (snr / 10)))
        m = evaluate(y, predict_proba(model, Xn.astype(np.float32), R, device))
        out["clean" if snr is None else f"{snr} dB"] = dict(macro_f1=m["macro_f1"], accuracy=m["accuracy"])
    return out


def uncertainty(prob, y):
    pred, conf = prob.argmax(1), prob.max(1)
    rows = []
    for thr in [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
        m = conf >= thr
        rows.append(dict(threshold=thr, coverage=float(m.mean()),
                         accuracy=float((pred[m] == y[m]).mean()) if m.any() else None))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--models", nargs="+", default=["cnn", "lstm", "transformer"])
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--quick", action="store_true", help="few records, few epochs")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    np.random.seed(a.seed)
    rng = np.random.default_rng(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_recs, test_recs, epochs = ep.DS1, ep.DS2, a.epochs
    if a.quick:
        train_recs, test_recs, epochs = ep.DS1[:8], ep.DS2[:6], min(4, a.epochs)
    tag = "quick" if a.quick else "full"
    Path("models").mkdir(exist_ok=True); Path("artifacts").mkdir(exist_ok=True)

    print(f"Device: {device} | run: {tag}")
    print("Building training set (DS1)...")
    Xa, Ra, ya, reca, _ = ep.build_dataset(train_recs, f"{a.data_dir}/mitdb", f"{a.data_dir}/cache_train_{tag}.npz")
    print("Building test set (DS2)...")
    Xt, Rt, yt, rect, post = ep.build_dataset(test_recs, f"{a.data_dir}/mitdb", f"{a.data_dir}/cache_test_{tag}.npz")

    strat = ya if np.bincount(ya).min() >= 2 else None
    Xtr, Xva, Rtr, Rva, ytr, yva = train_test_split(Xa, Ra, ya, test_size=0.1, random_state=a.seed, stratify=strat)
    print(f"train {len(Xtr)} | val {len(Xva)} | test {len(Xt)} beats")

    results = dict(meta=dict(protocol="inter-patient (DS1 train / DS2 test)", run=tag, epochs=epochs,
                             n_train=int(len(Xtr)), n_val=int(len(Xva)), n_test=int(len(Xt)),
                             train_records=[int(r) for r in train_recs],
                             test_records=[int(r) for r in test_recs], classes=ep.CLASSES,
                             class_counts_train={c: int((ya == i).sum()) for i, c in enumerate(ep.CLASSES)},
                             class_counts_test={c: int((yt == i).sum()) for i, c in enumerate(ep.CLASSES)}),
                   models={})

    for key in a.models:
        t0 = time.time()
        print(f"\nTraining {ep.MODEL_TITLES[key]}...")
        model, hist = train_model(key, Xtr, Rtr, ytr, Xva, Rva, yva, epochs, device, f"models/{key}.pt", a.seed)
        prob = predict_proba(model, Xt, Rt, device)
        m = evaluate(yt, prob)
        m.update(title=ep.MODEL_TITLES[key], history=hist, train_seconds=round(time.time() - t0, 1),
                 robustness=robustness(model, Xt, Rt, yt, device, rng), uncertainty=uncertainty(prob, yt))
        results["models"][key] = m
        print(f"  TEST accuracy {m['accuracy']:.4f} | macro-F1 {m['macro_f1']:.4f}")

        pred, conf = prob.argmax(1), prob.max(1)
        wrong = np.where(pred != yt)[0]
        pd.DataFrame(dict(record=rect[wrong], sample=post[wrong],
                          true=[ep.CLASSES[i] for i in yt[wrong]], pred=[ep.CLASSES[i] for i in pred[wrong]],
                          confidence=conf[wrong].round(4), prev_rr=Rt[wrong, 0].round(3),
                          next_rr=Rt[wrong, 1].round(3))).to_csv(f"artifacts/errors_{key}.csv", index=False)

    Path("artifacts/metrics.json").write_text(json.dumps(results, indent=2))
    print("\nSaved models/, artifacts/metrics.json and artifacts/errors_*.csv")

    try:
        from prepare_demo import prepare_demo
        print("\nPreparing demo recordings...")
        prepare_demo(a.data_dir)
    except Exception as exc:
        print("Demo preparation skipped:", exc)


if __name__ == "__main__":
    main()
