from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from .config import PipelineConfig
from .io import load_area_and_similarity, save_features
from .features import extract_hard_features, apply_rule_and_uncertainty
from .postprocess import median_smooth

@dataclass
class TSTPipeline:
    cfg: PipelineConfig

    def extract_features(self) -> pd.DataFrame:
        area, sim = load_area_and_similarity(self.cfg.paths.input_npy)
        df = extract_hard_features(
            area, sim,
            fps=self.cfg.window.fps,
            win_s=self.cfg.window.win_s,
            hop_s=self.cfg.window.hop_s,
            beta=self.cfg.rule.beta,
            w_min=self.cfg.rule.w_min,
            norm_within_video=self.cfg.norm_within_video,
            add_lowfreq_power=self.cfg.add_lowfreq_power,
        )
        df = apply_rule_and_uncertainty(df, threshold=self.cfg.rule.threshold, band=self.cfg.rule.band)
        save_features(df, self.cfg.paths.out_features_csv)
        return df

    def route_windows(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        easy = df[df['is_uncertain'] == 0].copy()
        hard = df[df['is_uncertain'] == 1].copy()
        return easy, hard

    def smooth_probs(self, probs: np.ndarray, k:int=7) -> np.ndarray:
        return median_smooth(probs, k=k)
