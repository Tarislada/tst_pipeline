from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

@dataclass
class VerifierConfig:
    feature_cols: list[str] | None = None
    n_estimators: int = 200
    max_depth: int | None = None
    random_state: int = 0

class VerifierRF:
    def __init__(self, cfg: VerifierConfig):
        self.cfg = cfg
        self.model = RandomForestClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            random_state=cfg.random_state,
            n_jobs=-1
        )

    def fit(self, df: pd.DataFrame, y: np.ndarray) -> None:
        X = df[self._features(df)]
        self.model.fit(X, y)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self._features(df)]
        return self.model.predict_proba(X)[:,1]

    def _features(self, df: pd.DataFrame) -> list[str]:
        if self.cfg.feature_cols is not None:
            return self.cfg.feature_cols
        # default set
        base = [
            "var_area_raw_norm","var_area_eff_norm","std_area_norm","range_area_norm",
            "sum_absdiff_area_norm","p80_absdiff_norm","rel_disp_area_norm",
            "mean_sim","std_sim","lf_pow_0_2hz"
        ]
        return [c for c in base if c in df.columns]
