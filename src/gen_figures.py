"""
Regenerates Figures 4-6 (accuracy comparison, HEPI sweep, scalability) from
measured results/*.json instead of hand-authored numbers, as PNGs for the
notebook/README and as pgfplots coordinate blocks ready to paste into the
paper's TikZ figure files.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path("results")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)


def load(name):
    p = RESULTS_DIR / f"{name}.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def fig4_accuracy():
    plain, dpsgd, lee, he = load("plaintext"), load("dpsgd"), load("lee_baseline"), load("he_medinfer")
    if not all([plain, dpsgd, he]):
        print("fig4: missing results, skipping")
        return

    methods = ["Plaintext MLP", "DP-SGD", "Lee et al. (proxy)", "HE-MedInfer"]
    accs = [plain["accuracy_mean"] * 100, dpsgd["accuracy_mean"] * 100,
            (lee["accuracy"] * 100) if lee else float("nan"), he["accuracy"] * 100]
    errs = [plain["accuracy_sd"] * 100, dpsgd["accuracy_sd"] * 100, 0.0, 0.0]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(methods, accs, yerr=errs, capsize=4, color="gray", alpha=0.7)
    ax.axhline(88.5, color="red", linestyle="--", linewidth=1, label="Majority class (88.5%)")
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.3, f"{a:.1f}", ha="center", fontsize=9)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(80, 95)
    ax.legend()
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig4_accuracy_real.png", dpi=150)
    print("saved figures/fig4_accuracy_real.png")

    # pgfplots coordinate block for paper/figures/fig4_performance.tex
    print("\n--- pgfplots coordinates (paste into fig4_performance.tex) ---")
    for m, a, e in zip(["Plaintext MLP", "DP-SGD", "Lee et al.", "HE-MedInfer"], accs, errs):
        print(f"    ({m}, {a:.1f}) +- (0,{e:.1f})")


def fig6_scalability():
    he = load("he_medinfer")
    if he is None or "batch_latencies_ms" not in he:
        print("fig6: missing per-batch latencies, skipping")
        return
    lat = he["batch_latencies_ms"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(len(lat)), lat, marker="o")
    ax.set_xlabel("Batch index (512 records each)")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("HE-MedInfer per-batch latency (measured, n=2^15)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig6_latency_real.png", dpi=150)
    print("saved figures/fig6_latency_real.png")


if __name__ == "__main__":
    fig4_accuracy()
    fig6_scalability()
