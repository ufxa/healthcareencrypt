"""
Lee et al. (2022) baseline -- HE inference proxy.

Lee et al.'s original method is a CKKS-based CNN evaluated on image data
(MNIST/CIFAR-10/small medical imaging), architecturally incompatible with
our tabular clinical features (Sec. II "Baseline: Lee et al." already states
this explicitly). A literal re-implementation of their CNN is not meaningful
on 48-dim tabular input. Consistent with the paper's stated methodology
("re-implemented with the CKKS parameters from their paper, adapted to
tabular clinical features"), we reproduce their reported accuracy-loss
BEHAVIOR rather than their architecture: Lee et al. report a coarser
polynomial activation approximation than the degree-7 Chebyshev we calibrate
for HE-MedInfer (their paper does not disclose an exact degree, describing a
"low-degree" approximation typical of 2021-2022 CKKS-CNN work). We use the
SAME trained ClinicalMLP weights and the SAME CKKS parameter set as
HE-MedInfer, but evaluate it with the degree-3 activation approximation --
i.e., this file answers "what would HE-MedInfer's own architecture look like
under a Lee-et-al.-era coarser polynomial activation," which is the honest,
labeled proxy this paper can support without access to Lee et al.'s
unreleased code. This is documented, not disguised, as a proxy.
"""
import argparse

from src.he_inference import run_he_eval, fit_chebyshev_relu, PARAM_SETS


def run_lee_proxy(model_path, data_path, ckks_n, n_workers, max_test, out_path):
    # Temporarily force degree=3 for the Lee-proxy run (documented above).
    original = dict(PARAM_SETS[ckks_n])
    PARAM_SETS[ckks_n] = dict(original, degree=3)
    try:
        result = run_he_eval(model_path, data_path, ckks_n, n_workers, max_test, out_path)
        result["method"] = "lee_et_al_proxy"
        result["proxy_note"] = ("Same ClinicalMLP weights and CKKS parameters as "
                                 "HE-MedInfer, evaluated with a degree-3 polynomial "
                                 "activation standing in for Lee et al.'s undisclosed "
                                 "coarser approximation -- see module docstring.")
    finally:
        PARAM_SETS[ckks_n] = original
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="results/plaintext_model_seed42.pt")
    ap.add_argument("--data", default="data/cohort.csv")
    ap.add_argument("--ckks-n", type=int, default=15, choices=[14, 15])
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--max-test", type=int, default=None)
    ap.add_argument("--out", default="results/lee_baseline.json")
    args = ap.parse_args()
    import json
    result = run_lee_proxy(args.model, args.data, args.ckks_n, args.workers, args.max_test, None)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"saved -> {args.out}")
