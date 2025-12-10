from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Iterator

def window_indices(n_frames:int, fps:int, win_s:float=1.0, hop_s:Optional[float]=None) -> Iterator[tuple[int,int]]:
    hop_s = hop_s or win_s
    win = int(round(win_s * fps))
    hop = int(round(hop_s * fps))
    for start in range(0, max(0, n_frames - win + 1), hop):
        yield start, start + win

def safe_minmax_norm(x, ref_min=None, ref_max=None, eps=1e-9):
    x = np.asarray(x, dtype=np.float32)
    mn = np.min(x) if ref_min is None else ref_min
    mx = np.max(x) if ref_max is None else ref_max
    den = (mx - mn) + eps
    return (x - mn) / den, mn, mx

def lowfreq_power(signal, fps, f_lo=0.0, f_hi=2.0):
    x = signal - np.mean(signal)
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=1.0/fps)
    band = (freqs >= f_lo) & (freqs <= f_hi)
    return float(np.sum(np.abs(spec[band])**2) / (len(x) + 1e-9))

def extract_hard_features(
    area: np.ndarray,
    sim: Optional[np.ndarray],
    fps: int = 30,
    win_s: float = 1.0,
    hop_s: Optional[float] = None,
    beta: float = 0.3,
    w_min: float = 0.5,
    norm_within_video: bool = True,
    add_lowfreq_power: bool = True
) -> pd.DataFrame:
    area = np.asarray(area, dtype=np.float32).reshape(-1)
    if sim is None:
        sim = np.zeros_like(area, dtype=np.float32)
    else:
        sim = np.clip(np.asarray(sim, dtype=np.float32).reshape(-1), 0.0, 1.0)
    assert len(area) == len(sim), "area and sim length mismatch"

    T = len(area)
    w = np.clip(1.0 - beta * sim, w_min, 1.0)

    rows = []
    for s, e in window_indices(T, fps=fps, win_s=win_s, hop_s=hop_s):
        a = area[s:e]
        wloc = w[s:e]
        sloc = sim[s:e]
        if len(a) == 0:
            continue

        mean_area = float(np.mean(a))
        std_area  = float(np.std(a))
        var_area_raw = float(np.var(a))
        range_area = float(np.max(a) - np.min(a))
        absdiff = np.abs(np.diff(a)) if len(a) > 1 else np.array([0.], dtype=np.float32)
        sum_absdiff_area = float(np.sum(absdiff))
        p80_absdiff = float(np.percentile(absdiff, 80)) if len(absdiff) else 0.0
        rel_disp_area = float(np.max(a) - np.min(a))
        var_area_eff = float(np.var(a * wloc))
        mean_sim = float(np.mean(sloc))
        std_sim  = float(np.std(sloc))
        lf_pow = lowfreq_power(np.diff(a) if len(a) > 1 else np.zeros(1, np.float32), fps) if add_lowfreq_power else 0.0

        rows.append(dict(
            t_start_sec=s/float(fps), t_end_sec=e/float(fps), n_frames=len(a),
            mean_area=mean_area, std_area=std_area, var_area_raw=var_area_raw,
            var_area_eff=var_area_eff, range_area=range_area,
            sum_absdiff_area=sum_absdiff_area, p80_absdiff=p80_absdiff,
            rel_disp_area=rel_disp_area, mean_sim=mean_sim, std_sim=std_sim,
            lf_pow_0_2hz=lf_pow
        ))

    df = pd.DataFrame(rows)
    if not df.empty and norm_within_video:
        for col in ["var_area_raw","var_area_eff","range_area","sum_absdiff_area","p80_absdiff","rel_disp_area","std_area"]:
            df[col+"_norm"], mn, mx = safe_minmax_norm(df[col].values)
    return df

def apply_rule_and_uncertainty(df: pd.DataFrame, threshold: float = 0.02, band: float = 0.30) -> pd.DataFrame:
    use_norm = "var_area_eff_norm" in df.columns
    x = df["var_area_eff_norm" if use_norm else "var_area_eff"].values
    state = (x >= threshold).astype(int)  # 0 immobile, 1 mobile
    lo, hi = threshold - band/2.0, threshold + band/2.0
    uncertain = (x > lo) & (x < hi)
    out = df.copy()
    out["rule_score"] = x
    out["state_rule"] = state
    out["is_uncertain"] = uncertain.astype(int)
    return out
