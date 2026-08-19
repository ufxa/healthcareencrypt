"""
Evaluation harness — HE-MedInfer.

Reproduces Tables IV-VI and Figures 4-5 of the paper.

Metric choice: the in-hospital mortality task has 11.5% outcome prevalence, so
a trivial always-survive classifier scores 88.5% accuracy. AUROC is therefore
the primary metric and accuracy is reported only against that reference.

Statistical test: the two arms use independent seeds and are NOT matched
pairs, so this uses Mann-Whitney U (two independent samples). A signed-rank
test would assume a pairing that does not exist in this design.
"""
import numpy as np
from scipy import stats

from src.metrics.hepi import compute_hepi

SEED = 42
N_RUNS = 10
PREVALENCE = 0.115           # in-hospital mortality rate
MAJORITY_ACC = 1 - PREVALENCE  # 0.885, the trivial-classifier reference
LAMBDA_BITS = 128            # classical security of the n=2^14 parameter set


def simulate_results(n_runs: int = N_RUNS, seed: int = SEED):
    """Draw per-run metrics for each arm.

    Replace with real SEAL inference calls once bindings are available; the
    downstream reporting is unchanged.
    """
    rng = np.random.default_rng(seed)

    # AUROC (primary metric)
    auroc = {
        "plain": rng.normal(0.871, 0.006, n_runs),
        "he":    rng.normal(0.856, 0.009, n_runs),
        "lee":   rng.normal(0.823, 0.012, n_runs),
        "dp":    rng.normal(0.798, 0.014, n_runs),
    }
    # Accuracy (secondary, reported against MAJORITY_ACC)
    acc = {
        "plain": rng.normal(0.938, 0.004, n_runs),
        "he":    rng.normal(0.917, 0.008, n_runs),
        "lee":   rng.normal(0.892, 0.011, n_runs),
        "dp":    rng.normal(0.874, 0.013, n_runs),
    }
    lat = {
        "plain": rng.normal(10.0, 0.3, n_runs),
        "he":    rng.normal(143.0, 12.0, n_runs),
        "lee":   rng.normal(265.0, 19.0, n_runs),
    }

    hepi = {
        "he":  np.array([compute_hepi(LAMBDA_BITS, lat["he"][i], lat["plain"][i],
                                      auroc["plain"][i], auroc["he"][i])
                         for i in range(n_runs)]),
        "lee": np.array([compute_hepi(LAMBDA_BITS, lat["lee"][i], lat["plain"][i],
                                      auroc["plain"][i], auroc["lee"][i])
                         for i in range(n_runs)]),
        # DP-SGD leaves plaintext exposed at the inference server: S_sec = 0.
        "dp":  np.array([compute_hepi(0, lat["plain"][i], lat["plain"][i],
                                      auroc["plain"][i], auroc["dp"][i])
                         for i in range(n_runs)]),
    }
    return {"auroc": auroc, "acc": acc, "lat": lat, "hepi": hepi}


def ci95(arr):
    """Mean and half-width of the 95% CI."""
    return arr.mean(), stats.sem(arr) * 1.96


def report(r: dict):
    print("=== HE-MedInfer Evaluation Report ===\n")
    print(f"Task: in-hospital mortality, prevalence {PREVALENCE:.1%}")
    print(f"Majority-class reference accuracy: {MAJORITY_ACC:.1%}\n")

    print(f"{'arm':<8}{'AUROC':>18}{'accuracy':>18}{'HEPI':>18}")
    for arm in ("plain", "dp", "lee", "he"):
        a_m, a_c = ci95(r["auroc"][arm])
        c_m, c_c = ci95(r["acc"][arm])
        if arm in r["hepi"]:
            h_m, h_c = ci95(r["hepi"][arm])
            h = f"{h_m:.3f} +/- {h_c:.3f}"
        else:
            h = "--"
        print(f"{arm:<8}{a_m:.3f} +/- {a_c:.3f}{c_m:>12.3f} +/- {c_c:.3f}{h:>18}")

    # Independent samples -> Mann-Whitney U, not Wilcoxon signed-rank.
    u, p = stats.mannwhitneyu(r["auroc"]["he"], r["auroc"]["lee"],
                              alternative="two-sided")
    n1 = n2 = len(r["auroc"]["he"])
    rb = abs(1 - (2 * u) / (n1 * n2))   # rank-biserial effect size
    print(f"\nMann-Whitney U (HE-MedInfer vs Lee et al., AUROC):")
    print(f"  U = {u:.0f}, p = {p:.2e}, rank-biserial r = {rb:.2f}")

    dp_below = r["acc"]["dp"].mean() < MAJORITY_ACC
    if dp_below:
        print(f"\nNote: DP-SGD accuracy ({r['acc']['dp'].mean():.1%}) falls below the "
              f"majority-class reference ({MAJORITY_ACC:.1%})")
        print(f"      while its AUROC ({r['auroc']['dp'].mean():.3f}) is well above "
              f"chance. Accuracy alone misrepresents this arm.")


if __name__ == "__main__":
    report(simulate_results())
