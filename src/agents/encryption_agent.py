"""
Encryption Agent (EA) — HE-MedInfer
Handles feature normalization, SIMD packing, and CKKS encryption.
"""
import numpy as np


def normalize(x: np.ndarray, low: float = -1.0, high: float = 1.0) -> np.ndarray:
    xmin, xmax = x.min(axis=0), x.max(axis=0)
    return low + (x - xmin) / (xmax - xmin + 1e-9) * (high - low)


def simd_pack(records: np.ndarray, slot_count: int) -> list:
    """Pack multiple records into CKKS plaintext slots."""
    batches = []
    for i in range(0, len(records), slot_count):
        batch = records[i : i + slot_count]
        batches.append(batch.flatten().tolist())
    return batches


def encrypt(plaintext: list, context) -> object:
    """Encrypt a packed plaintext with the provided SEAL/HEAAN context."""
    raise NotImplementedError("Wire up Microsoft SEAL Python bindings here.")
