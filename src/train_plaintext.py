"""
Trains the plaintext MLP (upper-bound baseline) on GPU.
Also serves as the reference model whose weights are later evaluated
homomorphically in he_inference.py — the *same* trained weights are used in
plaintext and under HE, so the HE-vs-plaintext AUROC gap in the paper is
measured on paired inputs/weights, not two independently trained models.
"""
import argparse
import json
import time

import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split

from src.datagen.preprocessing import load_and_preprocess
from src.models import ClinicalMLP


def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1] if i < n_bins - 1 else y_prob <= bins[i + 1])
        if mask.sum() == 0:
            continue
        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        ece += (mask.sum() / len(y_true)) * abs(conf - acc)
    return float(ece)


def train_one_run(X_tr_full, y_tr_full, X_te, y_te, device, seed, max_epochs=60,
                   lr=5e-4, weight_decay=3e-3, batch_size=256, patience=8):
    """
    Trains with early stopping on a held-out validation split (10% of train,
    disjoint from the test split) -- matches the paper's Sec. VII "Internal
    validity" claim. Positive-class weighting compensates for the 11.5%
    outcome prevalence; without it, Adam converges to a near-majority-class
    solution within a handful of epochs and AUROC degrades from there
    (verified empirically -- see scripts/debug_training.py).
    """
    torch.manual_seed(seed)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr_full, y_tr_full, test_size=0.10, random_state=seed, stratify=y_tr_full
    )

    model = ClinicalMLP(in_dim=X_tr.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    pos_weight = torch.tensor([(1 - y_tr.mean()) / y_tr.mean()], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    Xtr_t = torch.tensor(X_tr, device=device)
    ytr_t = torch.tensor(y_tr, device=device)
    Xval_t = torch.tensor(X_val, device=device)
    n = len(Xtr_t)

    best_auroc, best_state, best_epoch, bad_epochs = -1.0, None, 0, 0
    t0 = time.time()
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            logits = model(Xtr_t[idx])
            loss = loss_fn(logits, ytr_t[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_probs = torch.sigmoid(model(Xval_t)).cpu().numpy()
        val_auroc = roc_auc_score(y_val, val_probs)
        if val_auroc > best_auroc:
            best_auroc, best_epoch, bad_epochs = val_auroc, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
            best_val_probs = val_probs
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    model.load_state_dict(best_state)
    train_time = time.time() - t0

    # Decision threshold calibrated on the validation split by maximizing
    # validation accuracy (NOT the naive 0.5 cut, and not Youden's J -- the
    # latter balances TPR/FPR and produces a threshold far below the 11.5%
    # prevalence, which tanks accuracy well below the majority-class
    # reference; that is a real, known trade-off, but accuracy specifically
    # is reported here to be read against the majority-class baseline, so
    # its threshold is chosen accordingly). Threshold search happens on
    # validation only; applied to test without further tuning.
    grid = np.linspace(0.01, 0.99, 197)
    val_accs = [accuracy_score(y_val, (best_val_probs >= t).astype(int)) for t in grid]
    threshold = float(grid[int(np.argmax(val_accs))])

    model.eval()
    with torch.no_grad():
        Xte_t = torch.tensor(X_te, device=device)
        t0 = time.time()
        logits = model(Xte_t)
        infer_time_ms = (time.time() - t0) * 1000
        probs = torch.sigmoid(logits).cpu().numpy()

    auroc = roc_auc_score(y_te, probs)
    preds = (probs >= threshold).astype(int)
    acc = accuracy_score(y_te, preds)
    ece = expected_calibration_error(y_te, probs)

    return model, {
        "auroc": float(auroc), "accuracy": float(acc), "ece": ece,
        "threshold": threshold,
        "train_time_s": train_time, "best_epoch": best_epoch,
        "infer_time_ms_full_test": infer_time_ms,
        "infer_time_ms_per_512": infer_time_ms * 512 / len(X_te),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/cohort.csv")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="results/plaintext.json")
    ap.add_argument("--model-out", default="results/plaintext_model_seed42.pt")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    X_tr, X_te, y_tr, y_te, stats = load_and_preprocess(args.data)
    print(f"train={X_tr.shape} test={X_te.shape} prevalence(test)={stats['prevalence_test']:.4f}")

    metrics = []
    best_model = None
    for run in range(args.runs):
        seed = 42 + run
        model, m = train_one_run(X_tr, y_tr, X_te, y_te, device, seed)
        m["seed"] = seed
        metrics.append(m)
        print(f"run {run}: AUROC={m['auroc']:.4f} acc={m['accuracy']:.4f} "
              f"ece={m['ece']:.4f} train={m['train_time_s']:.1f}s")
        if run == 0:
            best_model = model  # seed=42 model is the one exported for HE inference

    torch.save(best_model.state_dict(), args.model_out)

    aurocs = [m["auroc"] for m in metrics]
    accs = [m["accuracy"] for m in metrics]
    result = {
        "method": "plaintext_mlp",
        "n_runs": args.runs,
        "auroc_mean": float(np.mean(aurocs)), "auroc_sd": float(np.std(aurocs, ddof=1)),
        "accuracy_mean": float(np.mean(accs)), "accuracy_sd": float(np.std(accs, ddof=1)),
        "ece_mean": float(np.mean([m["ece"] for m in metrics])),
        "runs": metrics,
        "preprocessing_stats": {k: v for k, v in stats.items() if k not in ("mu", "sigma")},
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"AUROC = {result['auroc_mean']:.4f} +/- {result['auroc_sd']:.4f}")
    print(f"saved -> {args.out}, model -> {args.model_out}")


if __name__ == "__main__":
    main()
