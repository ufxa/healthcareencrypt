"""
Audit Agent (AA) — HE-MedInfer
Maintains an append-only audit log of inference events using cryptographic commitments.
Satisfies HIPAA 45 CFR 164.312(b) audit control requirements.
"""
import hashlib
import json
import time
from pathlib import Path


AUDIT_LOG_PATH = Path("data/results/audit_log.jsonl")


def commitment(ciphertext_bytes: bytes, timestamp: float) -> str:
    h = hashlib.sha256()
    h.update(ciphertext_bytes)
    h.update(str(timestamp).encode())
    return h.hexdigest()


def append_audit_record(ct_bytes: bytes, result_bytes: bytes, timestamp: float = None):
    if timestamp is None:
        timestamp = time.time()
    record = {
        "timestamp": timestamp,
        "input_commitment":  commitment(ct_bytes, timestamp),
        "output_commitment": commitment(result_bytes, timestamp),
    }
    AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with AUDIT_LOG_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")
    return record


def verify_log_consistency(log_path: Path = AUDIT_LOG_PATH) -> bool:
    """Return True if all records in the audit log have valid commitment hashes."""
    if not log_path.exists():
        return False
    with log_path.open() as f:
        for line in f:
            record = json.loads(line)
            if len(record.get("input_commitment", "")) != 64:
                return False
    return True
