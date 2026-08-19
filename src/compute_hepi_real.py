"""
Aggregates the measured results/*.json files into the real HEPI table
(Table V of the paper) and a security-level table, replacing the
placeholder / hand-authored numbers used before the real pipeline existed.

Security level (S_sec, Eq. 2): rather than re-deriving lambda from scratch
via the Sage-based Lattice Estimator (a heavy dependency this repo does not
vendor), we use the modulus budget headroom against the published
HomomorphicEncryption.org / SEAL standard 128-bit-security thresholds for
each poly_modulus_degree -- the same standard table SEAL itself validates
parameters against. This is a documented simplification, stated as such in
the paper: it certifies "at least 128-bit" rather than computing an exact
continuous bit count.
"""
import json
from pathlib import Path

from src.metrics.hepi import compute_hepi

RESULTS_DIR = Path("results")

# HomomorphicEncryption.org / SEAL standard 128-bit-security max total
# coefficient-modulus bits, by poly_modulus_degree (the same table SEAL
# uses internally to validate parameters).
STD_128BIT_MAX_BITS = {8192: 218, 16384: 438, 32768: 881}

LAMBDA_TARGET = 128.0


def security_bits(poly_modulus_degree, used_bits):
    """
    Certifies >= 128-bit security whenever used_bits <= the standard
    threshold for that ring degree (this is exactly SEAL's own admission
    check -- it refuses to build a context otherwise). We do not claim a
    tighter estimate than that: lambda is reported as exactly 128 when the
    configuration is certified, reflecting a floor bound, not a precise
    reduction cost. Configurations that failed to build (see
    he_inference.py PARAM_SETS docstring) are not reported here at all.
    """
    max_bits = STD_128BIT_MAX_BITS[poly_modulus_degree]
    assert used_bits <= max_bits, (
        f"used {used_bits} bits exceeds the 128-bit standard threshold "
        f"{max_bits} for n={poly_modulus_degree} -- this configuration is "
        f"NOT certified at 128-bit security and must not be reported as such"
    )
    return LAMBDA_TARGET


def load(name):
    p = RESULTS_DIR / f"{name}.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def main():
    plain = load("plaintext")
    dpsgd = load("dpsgd")
    he = load("he_medinfer")
    lee = load("lee_baseline")

    if plain is None:
        raise SystemExit("results/plaintext.json missing -- run train_plaintext.py first")

    a_plain = plain["auroc_mean"]
    l_plain = plain["runs"][0]["infer_time_ms_per_512"]

    rows = []

    if he is not None:
        lam = security_bits(32768, sum([60] + [40] * 16 + [60]))
        h = compute_hepi(lam, he["latency_ms_per_batch_mean"], l_plain, a_plain, he["auroc"])
        rows.append(("HE-MedInfer (n=2^15, deg=7)", he["auroc"], he["accuracy"],
                      he["latency_ms_per_batch_mean"], lam, h))

    if lee is not None:
        lam = security_bits(32768, sum([60] + [40] * 16 + [60]))
        h = compute_hepi(lam, lee["latency_ms_per_batch_mean"], l_plain, a_plain, lee["auroc"])
        rows.append(("Lee et al. proxy (n=2^15, deg=3)", lee["auroc"], lee["accuracy"],
                      lee["latency_ms_per_batch_mean"], lam, h))

    if dpsgd is not None:
        # Non-HE: plaintext exposed at the inference server -> S_sec = 0 by
        # construction (Sec. III-C).
        h = compute_hepi(0.0, l_plain, l_plain, a_plain, dpsgd["auroc_mean"])
        rows.append(("DP-SGD (non-HE)", dpsgd["auroc_mean"], dpsgd["accuracy_mean"],
                      l_plain, 0.0, h))

    rows.append(("Plaintext MLP (upper bound)", a_plain, plain["accuracy_mean"],
                  l_plain, None, None))

    print(f"{'Method':<34}{'AUROC':>8}{'Acc':>8}{'Latency(ms)':>14}{'lambda':>8}{'HEPI':>8}")
    for name, auroc, acc, lat, lam, h in rows:
        lam_s = f"{lam:.0f}" if lam is not None else "--"
        h_s = f"{h:.3f}" if h is not None else "--"
        print(f"{name:<34}{auroc:>8.4f}{acc:>8.4f}{lat:>14.1f}{lam_s:>8}{h_s:>8}")

    out = {"rows": [{"method": r[0], "auroc": r[1], "accuracy": r[2],
                      "latency_ms": r[3], "lambda_bits": r[4], "hepi": r[5]}
                     for r in rows]}
    with open(RESULTS_DIR / "hepi_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {RESULTS_DIR / 'hepi_summary.json'}")


if __name__ == "__main__":
    main()
