"""
Trains the DP-SGD baseline (Opacus) with (epsilon=1.0, delta=1e-5)-DP,
same architecture as the plaintext MLP, for comparison in Table V.
"""
import argparse
import copy
import json
import time

import numpy as np
import torch
import torch.nn as nn
from opacus import PrivacyEngine
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

from src.datagen.preprocessing import load_and_preprocess
from src.models import ClinicalMLP
from src.train_plaintext import expected_calibration_error


def train_one_run_dp(X_tr_full, y_tr_full, X_te, y_te, device, seed,
                      epsilon=1.0, delta=1e-5, epochs=20, lr=1e-3,
                      batch_size=256, max_grad_norm=1.0):
    torch.manual_seed(seed)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr_full, y_tr_full, test_size=0.10, random_state=seed, stratify=y_tr_full
    )

    model = ClinicalMLP(in_dim=X_tr.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    pos_weight = torch.tensor([(1 - y_tr.mean()) / y_tr.mean()], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    privacy_engine = PrivacyEngine()
    model, opt, dl = privacy_engine.make_private_with_epsilon(
        module=model, optimizer=opt, data_loader=dl,
        target_epsilon=epsilon, target_delta=delta, epochs=epochs,
        max_grad_norm=max_grad_norm,
    )

    Xval_t = torch.tensor(X_val, device=device)
    best_auroc, best_state, best_epoch = -1.0, None, 0
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_probs = torch.sigmoid(model(Xval_t)).cpu().numpy()
        val_auroc = roc_auc_score(y_val, val_probs)
        if val_auroc > best_auroc:
            best_auroc, best_epoch = val_auroc, epoch
            best_state = copy.deepcopy(model.state_dict())
            best_val_probs = val_probs
    train_time = time.time() - t0
    actual_epsilon = privacy_engine.get_epsilon(delta)

    model.load_state_dict(best_state)
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

    return {
        "auroc": float(auroc), "accuracy": float(acc), "ece": ece,
        "threshold": threshold, "epsilon_target": epsilon,
        "epsilon_actual": float(actual_epsilon), "delta": delta,
        "train_time_s": train_time, "best_epoch": best_epoch,
        "infer_time_ms_full_test": infer_time_ms,
        "infer_time_ms_per_512": infer_time_ms * 512 / len(X_te),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/cohort.csv")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epsilon", type=float, default=1.0)
    ap.add_argument("--out", default="results/dpsgd.json")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    X_tr, X_te, y_tr, y_te, stats = load_and_preprocess(args.data)
    print(f"train={X_tr.shape} test={X_te.shape}")

    metrics = []
    for run in range(args.runs):
        seed = 42 + run
        m = train_one_run_dp(X_tr, y_tr, X_te, y_te, device, seed, epsilon=args.epsilon)
        m["seed"] = seed
        metrics.append(m)
        print(f"run {run}: AUROC={m['auroc']:.4f} acc={m['accuracy']:.4f} "
              f"eps_actual={m['epsilon_actual']:.3f} train={m['train_time_s']:.1f}s")

    aurocs = [m["auroc"] for m in metrics]
    accs = [m["accuracy"] for m in metrics]
    result = {
        "method": "dpsgd", "n_runs": args.runs,
        "auroc_mean": float(np.mean(aurocs)), "auroc_sd": float(np.std(aurocs, ddof=1)),
        "accuracy_mean": float(np.mean(accs)), "accuracy_sd": float(np.std(accs, ddof=1)),
        "ece_mean": float(np.mean([m["ece"] for m in metrics])),
        "epsilon_target": args.epsilon,
        "epsilon_actual_mean": float(np.mean([m["epsilon_actual"] for m in metrics])),
        "runs": metrics,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"AUROC = {result['auroc_mean']:.4f} +/- {result['auroc_sd']:.4f}")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
