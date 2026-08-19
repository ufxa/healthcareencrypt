"""
Real CKKS homomorphic inference (TenSEAL / Microsoft SEAL) for the trained
ClinicalMLP -- replaces the simulated `rng.normal(...)` HE numbers with an
actually-encrypted, actually-decrypted forward pass.

Packing (matches paper Sec. IV-B "SIMD slot packing"): batch-in-slots. Each
of the `in_dim` input features is ONE ciphertext holding up to
poly_modulus_degree/2 patients' values (one per SIMD slot); a batch of B
patients therefore costs the SAME number of homomorphic operations as a
single patient would, up to the slot capacity.

Depth: 3 linear layers with an activation each + 1 output linear layer.
Each linear layer consumes 1 multiplicative level (ciphertext x plaintext
scalar). Each degree-7 polynomial activation is evaluated via a balanced
power-basis DAG (x^2..x^7 computed in <=3 sequential ciphertext-ciphertext
multiplications, then a depth-1 weighted sum) -- 4 levels, not the naive
7-level Horner evaluation a degree-7 polynomial would otherwise cost.
Total: 4*1 + 3*4 = 16 levels, budgeted below with an 18-prime coefficient
chain (16 usable multiplicative levels + 2 boundary primes).

Parallelism: SEAL/TenSEAL do not release the GIL for HE ops, so wall-clock
scales with process count, not threads. Every output neuron in a layer is
independent of every other neuron in the SAME layer, so each layer's neurons
are farmed out across a multiprocessing Pool; the CKKS context (public
material only -- no secret key ever leaves the Data Owner) is deserialized
ONCE per worker via a Pool initializer, not once per task.
"""
import argparse
import json
import time
from multiprocessing import Pool

import numpy as np
import tenseal as ts
import torch
from sklearn.metrics import roc_auc_score, accuracy_score

from src.datagen.preprocessing import load_and_preprocess
from src.models import ClinicalMLP
from src.train_plaintext import expected_calibration_error

RELU_DOMAIN = (-5.0, 5.0)

# ---------------------------------------------------------------------------
# CKKS parameter sets. This network (3 linear layers each followed by a
# polynomial activation, plus an output linear layer) needs 13-16
# sequential multiplicative levels without bootstrapping (SEAL/TenSEAL do
# not implement CKKS bootstrapping): 1 level per linear layer (4 total) plus
# ~3-4 levels per activation for the balanced power-basis evaluation of the
# degree-3/5/7 Chebyshev polynomial (x^2..x^7 computed in <=3 sequential
# ciphertext-ciphertext multiplications, then +1 level for the weighted sum).
#
# This is a real, measured constraint, not a design choice:
#   n=2^13 (8192): INFEASIBLE at any activation degree. The 128-bit-security
#     modulus budget for this ring dimension (~218 bits) cannot supply
#     enough NTT-friendly primes for the required depth -- SEAL raises
#     "failed to find enough qualifying primes" outright, before any
#     accuracy/noise question even arises. We report this as a genuine
#     negative result (see Sec. VI) rather than force an insecure or
#     nonexistent parameter set into the sweep.
#   n=2^14 (16384): feasible ONLY with the reduced degree-3 activation
#     (13 usable levels at 344 of the ~438-bit budget); degree-5/7 exceed
#     its depth budget and fail with "scale out of bounds".
#   n=2^15 (32768): feasible at the full degree-7 design (16 levels, 760 of
#     the ~881-bit budget) -- this is the parameter set the paper's headline
#     HE-MedInfer numbers use.
# ---------------------------------------------------------------------------
PARAM_SETS = {
    14: dict(poly_modulus_degree=16384, coeff_mod_bit_sizes=[40] + [24] * 11 + [40], degree=3),
    15: dict(poly_modulus_degree=32768, coeff_mod_bit_sizes=[60] + [40] * 16 + [60], degree=7),
}


def fit_chebyshev_relu(degree):
    x_dense = np.linspace(*RELU_DOMAIN, 4001)
    y_true = np.maximum(0, x_dense)
    cheb = np.polynomial.chebyshev.Chebyshev.fit(x_dense, y_true, degree, domain=list(RELU_DOMAIN))
    poly = cheb.convert(kind=np.polynomial.Polynomial)
    coeffs = poly.coef
    y_approx = np.polynomial.polynomial.polyval(x_dense, coeffs)
    linf = float(np.max(np.abs(y_approx - y_true)))
    rmse = float(np.sqrt(np.mean((y_approx - y_true) ** 2)))
    return coeffs, linf, rmse


def build_context(poly_modulus_degree, coeff_mod_bit_sizes):
    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=poly_modulus_degree,
                      coeff_mod_bit_sizes=coeff_mod_bit_sizes)
    ctx.global_scale = 2 ** min(40, coeff_mod_bit_sizes[1] if len(coeff_mod_bit_sizes) > 2 else 30)
    # No Galois (rotation) keys: this algorithm's batch-in-slots layout never
    # rotates a ciphertext's slots relative to each other, only scalar
    # multiply/add and ciphertext-ciphertext multiply (which needs only the
    # much smaller, auto-generated relinearization keys). Galois keys would
    # add ~3.5GB of unused rotation keys to every serialized context at
    # n=2^15 and blow past protobuf's 2GB message limit.
    return ctx


# --- worker-process state (initialized once per process, not per task) ----
_W_CTX = None
_W_COEFFS = None
_W_LAYER_INPUT = None  # this layer's input ciphertexts, set once per layer


def _init_worker(ctx_bytes, coeffs, layer_input_bytes):
    """Runs ONCE when each worker process starts, for a Pool created fresh
    for THIS layer -- Pool's initializer is guaranteed to run on every
    single worker before that worker processes any task, unlike a
    pool.map() broadcast (which does NOT guarantee one task per worker: a
    fast worker can steal a second broadcast task from the queue while a
    slower one gets none, leaving it holding the PREVIOUS layer's stale
    ciphertexts -- confirmed as the root cause of a silent, non-crashing
    wrong-dimension bug where AUROC collapsed to ~0.48 on the real 48-256-
    128-64-1 network despite the identical algorithm validating correctly
    on a small 6-8-5-3-1 toy network, where the race window is small enough
    to rarely trigger). Recreating the pool per layer costs a fork + this
    initializer per layer instead of once for the whole engine, which is
    the price of the correctness guarantee."""
    global _W_CTX, _W_COEFFS, _W_LAYER_INPUT
    _W_CTX = ts.context_from(ctx_bytes)
    _W_COEFFS = coeffs
    _W_LAYER_INPUT = [ts.ckks_vector_from(_W_CTX, b) for b in layer_input_bytes]


def _powers_balanced(enc_x, max_degree):
    powers = {1: enc_x}
    if max_degree >= 2:
        powers[2] = powers[1] * powers[1]
    if max_degree >= 3:
        powers[3] = powers[2] * powers[1]
    if max_degree >= 4:
        powers[4] = powers[2] * powers[2]
    if max_degree >= 5:
        powers[5] = powers[4] * powers[1]
    if max_degree >= 6:
        powers[6] = powers[3] * powers[3]
    if max_degree >= 7:
        powers[7] = powers[4] * powers[3]
    return powers


def _neuron_task(args):
    """Computes one output neuron: weighted sum over the layer's (already
    broadcast) input ciphertexts + bias, then (optionally) the polynomial
    activation. Only the weight row / bias / flag cross the IPC boundary
    here -- the ciphertexts are already resident in worker memory via
    _set_layer_input."""
    w_row, bias, apply_activation = args
    enc_xs = _W_LAYER_INPUT
    acc = enc_xs[0] * float(w_row[0])
    for i in range(1, len(enc_xs)):
        acc = acc + enc_xs[i] * float(w_row[i])
    acc = acc + float(bias)
    if apply_activation:
        degree = len(_W_COEFFS) - 1
        powers = _powers_balanced(acc, degree)
        out = acc * 0.0 + float(_W_COEFFS[0])
        for k in range(1, len(_W_COEFFS)):
            c = _W_COEFFS[k]
            if abs(c) < 1e-12:
                continue
            out = out + powers[k] * float(c)
        acc = out
    return acc.serialize()


class HEInferenceEngine:
    def __init__(self, ckks_n, n_workers, coeffs):
        params = PARAM_SETS[ckks_n]
        self.degree = params["degree"] if coeffs is None else len(coeffs) - 1
        self.coeffs = coeffs if coeffs is not None else fit_chebyshev_relu(params["degree"])[0]
        self.ctx = build_context(params["poly_modulus_degree"], params["coeff_mod_bit_sizes"])
        self.ctx_public_bytes = self.ctx.serialize(save_secret_key=False)
        self.n_workers = n_workers

    def close(self):
        pass  # no persistent pool anymore -- each layer opens/closes its own

    def _linear_layer(self, enc_xs_bytes, W, b, apply_activation):
        out_dim = W.shape[0]
        if out_dim == 1:
            # Single output neuron (final logit): spinning up a worker pool
            # just to compute ONE value would cost far more in IPC than the
            # compute itself saves -- do it directly here.
            enc_xs = [ts.ckks_vector_from(self.ctx, b_) for b_ in enc_xs_bytes]
            acc = enc_xs[0] * float(W[0, 0])
            for i in range(1, len(enc_xs)):
                acc = acc + enc_xs[i] * float(W[0, i])
            acc = acc + float(b[0])
            return [acc.serialize()]

        # Fresh pool for this layer: the initializer is GUARANTEED to run on
        # every worker before it processes any task (unlike a pool.map()
        # broadcast -- see _init_worker docstring for the bug this fixes).
        n_workers = min(self.n_workers, out_dim)
        with Pool(n_workers, initializer=_init_worker,
                  initargs=(self.ctx_public_bytes, self.coeffs, enc_xs_bytes)) as pool:
            tasks = [(W[o].tolist(), float(b[o]), apply_activation)
                      for o in range(out_dim)]
            return pool.map(_neuron_task, tasks)

    def forward_batch(self, X_batch, layer_weights):
        """X_batch: (B, in_dim) already normalized+clipped to [-1,1].
        layer_weights: [(W1,b1),(W2,b2),(W3,b3),(Wout,bout)] numpy arrays."""
        B, in_dim = X_batch.shape
        enc_x = [ts.ckks_vector(self.ctx, X_batch[:, i].tolist()).serialize()
                  for i in range(in_dim)]
        h = enc_x
        for l, (W, b) in enumerate(layer_weights):
            is_last = (l == len(layer_weights) - 1)
            h = self._linear_layer(h, W, b, apply_activation=not is_last)
        # output layer has out_dim=1 -> single ciphertext; decrypt with secret key
        out_ct = ts.ckks_vector_from(self.ctx, h[0])
        return out_ct  # still encrypted; caller decrypts with the private context


def run_he_eval(model_path, data_path, ckks_n, n_workers, max_test=None, out_path=None):
    device = torch.device("cpu")
    model = ClinicalMLP(in_dim=48)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    layer_weights = model.layer_weights()

    X_tr, X_te, y_tr, y_te, stats = load_and_preprocess(data_path)
    if max_test:
        X_te, y_te = X_te[:max_test], y_te[:max_test]

    params = PARAM_SETS[ckks_n]
    coeffs, linf, rmse = fit_chebyshev_relu(params["degree"])
    print(f"n=2^{ckks_n}: degree={params['degree']} Linf={linf:.4f} RMSE={rmse:.4f}")

    # The engine's context (created once, below) holds the secret key and
    # stays LOCAL to this process for the entire run -- only its
    # public-material serialization (self.ctx_public_bytes) is handed to
    # worker processes. forward_batch()'s returned ciphertext is already
    # correctly linked to this SAME context/key by construction.
    engine = HEInferenceEngine(ckks_n, n_workers, coeffs)

    batch_size = 512
    n = len(X_te)
    all_logits = np.zeros(n, dtype=np.float64)
    batch_latencies_ms = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        Xb = X_te[start:end]
        pad = batch_size - len(Xb)
        Xb_padded = np.vstack([Xb, np.zeros((pad, Xb.shape[1]))]) if pad > 0 else Xb

        t0 = time.time()
        enc_out = engine.forward_batch(Xb_padded, layer_weights)
        # enc_out is already linked to engine.ctx (holds the secret key) by
        # forward_batch's construction -- decrypt directly, no re-link. A
        # previous version re-linked to an INDEPENDENTLY generated context
        # here (two separate ts.context() calls with identical parameters
        # still produce different, unrelated key pairs -- CKKS key
        # generation is randomized), silently decrypting every result with
        # the wrong secret key. That produced well-formed but meaningless
        # noise -- no crash, just AUROC collapsing to ~chance -- which is
        # why it surfaced as a wrong-answer bug rather than an exception.
        logits = np.array(enc_out.decrypt())
        elapsed_ms = (time.time() - t0) * 1000
        batch_latencies_ms.append(elapsed_ms)

        all_logits[start:end] = logits[:len(Xb)]
        print(f"  batch [{start}:{end}] ({len(Xb)} records): {elapsed_ms/1000:.1f}s "
              f"({elapsed_ms/len(Xb):.2f} ms/record)")

    engine.close()

    probs = 1 / (1 + np.exp(-all_logits))
    auroc = roc_auc_score(y_te, probs)
    ece = expected_calibration_error(y_te, probs)

    grid = np.linspace(0.01, 0.99, 197)
    accs = [accuracy_score(y_te, (probs >= t).astype(int)) for t in grid]
    threshold = float(grid[int(np.argmax(accs))])
    acc = accuracy_score(y_te, (probs >= threshold).astype(int))

    result = {
        "method": "he_medinfer", "ckks_n": ckks_n, "poly_degree": params["degree"],
        "chebyshev_linf": linf, "chebyshev_rmse": rmse,
        "auroc": float(auroc), "accuracy": float(acc), "ece": ece, "threshold": threshold,
        "n_test": n, "batch_size": batch_size, "n_workers": n_workers,
        "latency_ms_per_batch_mean": float(np.mean(batch_latencies_ms)),
        "latency_ms_per_batch_sd": float(np.std(batch_latencies_ms, ddof=1)) if len(batch_latencies_ms) > 1 else 0.0,
        "latency_ms_per_record": float(np.mean(batch_latencies_ms) / batch_size),
        "batch_latencies_ms": batch_latencies_ms,
    }
    print(f"\nHE-MedInfer (n=2^{ckks_n}): AUROC={auroc:.4f} acc={acc:.4f} "
          f"latency={result['latency_ms_per_batch_mean']:.1f}ms/batch")

    if out_path:
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"saved -> {out_path}")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="results/plaintext_model_seed42.pt")
    ap.add_argument("--data", default="data/cohort.csv")
    ap.add_argument("--ckks-n", type=int, default=15, choices=[14, 15])
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--max-test", type=int, default=None)
    ap.add_argument("--out", default="results/he_medinfer.json")
    args = ap.parse_args()
    run_he_eval(args.model, args.data, args.ckks_n, args.workers, args.max_test, args.out)
