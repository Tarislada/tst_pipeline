from __future__ import annotations
import numpy as np
import pandas as pd

def load_area_and_similarity(npy_path: str) -> tuple[np.ndarray, np.ndarray | None]:
    arr = np.load(npy_path)
    if arr.ndim == 1:
        area, sim = arr.astype('float32'), None
    else:
        area = arr[:,0].astype('float32')
        sim  = arr[:,1].astype('float32')
    return area, sim

def save_features(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)
