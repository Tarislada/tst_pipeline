from __future__ import annotations
import numpy as np

def median_smooth(probs: np.ndarray, k: int = 7) -> np.ndarray:
    k = max(1, k | 1)  # make odd
    pad = k // 2
    x = np.pad(probs, (pad, pad), mode='edge')
    out = np.empty_like(probs)
    for i in range(len(probs)):
        out[i] = np.median(x[i:i+k])
    return out
