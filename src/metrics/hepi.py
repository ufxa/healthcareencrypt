"""
HE Privacy Index (HEPI) — Novel metric for HE-MedInfer.
Eq. (1) from the paper: composite of security, overhead, and utility.

    HEPI = alpha * S_sec + beta * O_HE + gamma * Q_acc

    S_sec = min(1, lambda / 128)                       Eq. (2)
    O_HE  = 1 / (1 + log10(L_HE / L_plain))            Eq. (3)
    Q_acc = 1 - |A_plain - A_HE| / A_plain             Eq. (4)

Two definitions changed relative to the first draft of this metric, and both
changes are load-bearing:

1. S_sec is derived from the *bits of classical security* of the RLWE
   parameter set (Lattice Estimator), NOT from the residual CKKS noise
   budget. The noise budget measures remaining multiplicative depth before
   decryption failure; it says nothing about lattice hardness. A ciphertext
   with an exhausted noise budget is unusable, not insecure.

2. O_HE is log-scaled. The earlier linear form max(0, 1 - L_HE/L_plain) goes
   negative for any overhead beyond 2x and therefore clipped to zero across
   the entire operating range of every practical HE scheme, silently deleting
   the overhead dimension from the composite. With the reported 14.3x
   overhead the old formula capped HEPI at 0.791, making the reported 0.883
   unreachable.
"""

import math

LAMBDA_TARGET = 128  # bits of classical security targeted by deployment


def security_score(lambda_bits: float, lambda_target: int = LAMBDA_TARGET) -> float:
    """S_sec (Eq. 2): attained classical security normalized to the target.

    lambda_bits comes from the Lattice Estimator for the CKKS parameter set
    (n, q, chi), taking the minimum over primal-uSVP and dual attack models.
    Non-HE methods expose plaintext to the inference server: pass 0.0.
    """
    if lambda_bits < 0:
        raise ValueError("lambda_bits must be non-negative")
    return min(1.0, lambda_bits / lambda_target)


def overhead_score(latency_he: float, latency_plain: float) -> float:
    """O_HE (Eq. 3): log-scaled latency penalty, strictly positive.

    Maps 1x overhead -> 1.00, 10x -> 0.50, 100x -> 0.33, 1000x -> 0.25.
    """
    if latency_plain <= 0 or latency_he <= 0:
        raise ValueError("latencies must be positive")
    ratio = max(1.0, latency_he / latency_plain)  # HE is never faster
    return 1.0 / (1.0 + math.log10(ratio))


def utility_score(a_plain: float, a_he: float) -> float:
    """Q_acc (Eq. 4): retained discriminative performance.

    `a_plain` / `a_he` are AUROC, not accuracy: the in-hospital mortality
    task has 11.5% prevalence, so a trivial classifier scores 88.5% accuracy
    and accuracy differences are close to uninformative.
    """
    if not 0.0 < a_plain <= 1.0:
        raise ValueError("a_plain must be in (0, 1]")
    return 1.0 - abs(a_plain - a_he) / a_plain


def compute_hepi(
    lambda_bits: float,
    latency_he: float,
    latency_plain: float,
    auroc_plain: float,
    auroc_he: float,
    alpha: float = 0.4,
    beta: float = 0.2,
    gamma: float = 0.4,
) -> float:
    """Compute the HE Privacy Index (Eq. 1). Returns a float in [0, 1].

    >>> round(compute_hepi(128, 143, 10, 0.871, 0.856), 3)   # Table V, ours
    0.886
    >>> round(compute_hepi(128, 265, 10, 0.871, 0.823), 3)   # Lee et al.
    0.86
    >>> round(compute_hepi(0, 10, 10, 0.871, 0.798), 3)      # DP-SGD, non-HE
    0.566
    """
    if abs(alpha + beta + gamma - 1.0) > 1e-9:
        raise ValueError("weights must sum to 1")
    if min(alpha, beta, gamma) < 0:
        raise ValueError("weights must be non-negative")

    return (
        alpha * security_score(lambda_bits)
        + beta * overhead_score(latency_he, latency_plain)
        + gamma * utility_score(auroc_plain, auroc_he)
    )


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=False)

    # Reproduces Table IV and Fig. 5 of the paper.
    print(f"{'configuration':<26}{'lambda':>7}{'L_HE':>7}{'AUROC':>8}{'HEPI':>8}")
    for name, lam, lat, auroc in [
        ("HE-MedInfer n=2^13", 109, 78, 0.841),
        ("HE-MedInfer n=2^14", 128, 143, 0.856),
        ("HE-MedInfer n=2^15", 128, 412, 0.858),
        ("Lee et al.  n=2^14", 128, 265, 0.823),
        ("DP-SGD (non-HE)", 0, 10, 0.798),
    ]:
        h = compute_hepi(lam, lat, 10.0, 0.871, auroc)
        print(f"{name:<26}{lam:>7}{lat:>7}{auroc:>8.3f}{h:>8.3f}")
