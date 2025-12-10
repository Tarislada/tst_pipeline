from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class WindowConfig:
    fps: int = 30
    win_s: float = 1.0
    hop_s: float | None = None

@dataclass
class RuleConfig:
    beta: float = 0.31
    w_min: float = 0.75
    threshold: float = 0.02
    band: float = 0.30  # width of 'uncertain' band around threshold

@dataclass
class Paths:
    input_npy: str = "output_data.npy"   # shape (T,) area or (T,2) [area, sim]
    out_features_csv: str = "tst_features_persec.csv"

@dataclass
class PipelineConfig:
    window: WindowConfig = field(default_factory=WindowConfig)
    rule: RuleConfig = field(default_factory=RuleConfig)
    paths: Paths = field(default_factory=Paths)
    norm_within_video: bool = True
    add_lowfreq_power: bool = True
