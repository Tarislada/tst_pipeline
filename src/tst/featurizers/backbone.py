from __future__ import annotations
import numpy as np
import torch

class YOLOBackboneFeaturizer:
    """Placeholder interface for extracting per-frame backbone features from a running YOLO-seg model.
    Implement `extract_per_frame` to return a (T, C) or (T, C, H, W) tensor per video, then pool temporally.
    """
    def __init__(self, model=None, device: str = 'cuda'):
        self.model = model
        self.device = device

    def extract_per_frame(self, images: list[np.ndarray]) -> np.ndarray:
        """Return per-frame feature vectors for each frame image (list of HxWx3 uint8 arrays)."""
        raise NotImplementedError

    def temporal_pool_1s(self, feats_per_frame: np.ndarray, fps: int, sub_bins: int = 5) -> np.ndarray:
        """Pool features within each 1s window, optionally using sub-bins (e.g., 5×200ms)."""
        T, C = feats_per_frame.shape[:2]
        win = fps
        if T < win:
            return np.empty((0, C), dtype=np.float32)
        pooled = []
        for start in range(0, T - win + 1, win):
            chunk = feats_per_frame[start:start+win]  # (win, C)
            pooled.append(chunk.mean(axis=0))
        return np.stack(pooled, axis=0).astype(np.float32)
