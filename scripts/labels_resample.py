import os, re, glob
import numpy as np
import pandas as pd
import typer
from rich import print
from typing import Optional, Dict, List, Tuple

app = typer.Typer(help="Convert frame-level labels to per-second labels and rename to match feature bases.")

# ---------- utility ----------

def _ensure_dir(path: str):
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)

def _read_label_series(path: str) -> np.ndarray:
    """Accept CSVs that either have a 'label' column or a single numeric/bool column."""
    df = pd.read_csv(path)
    if "label" in df.columns:
        y = df["label"].to_numpy()
    else:
        num = df.select_dtypes(include=[np.number, "bool"]).columns.tolist()
        if not num:
            raise ValueError(f"{path}: no numeric/bool column found for labels.")
        y = df[num[0]].to_numpy()
    y = np.asarray(y).astype(int).reshape(-1)
    uniq = set(np.unique(y).tolist())
    if not uniq.issubset({0, 1}):
        raise ValueError(f"{path}: labels must be 0/1, got {sorted(uniq)}")
    return y

def _despike_by_duration(y: np.ndarray, fps: float, min_immobility_s: float) -> np.ndarray:
    """Remove label==1 runs shorter than min_immobility_s (in frames)."""
    y = y.astype(int, copy=True)
    n = len(y)
    thr = int(round(fps * float(min_immobility_s)))
    if n == 0 or thr <= 1:
        return y
    vals, lens, starts = [], [], []
    i = 0
    while i < n:
        j = i + 1
        while j < n and y[j] == y[i]:
            j += 1
        vals.append(y[i]); lens.append(j - i); starts.append(i)
        i = j
    y2 = y.copy()
    for v, L, s in zip(vals, lens, starts):
        if v == 1 and L < thr:
            y2[s:s+L] = 0
    return y2

def _downsample_to_seconds(y: np.ndarray, fps: float, rule: str = "majority") -> pd.DataFrame:
    """Return DataFrame with columns ['t_start_sec','label'], length floor(n_frames/fps)."""
    fps_int = int(round(float(fps)))
    if abs(float(fps) - fps_int) > 1e-3:
        print(f"[yellow]WARN[/yellow] non-integer fps={fps:.3f}; rounding to {fps_int}")
    fps = fps_int
    n_sec = max(0, len(y) // fps)
    if n_sec == 0:
        return pd.DataFrame(columns=["t_start_sec", "label"], dtype=int)
    y = y[: n_sec * fps].reshape(n_sec, fps)
    if rule == "any":
        sec = (y.sum(axis=1) > 0).astype(int)
    else:  # 'majority' default
        sec = (y.mean(axis=1) >= 0.5).astype(int)
    return pd.DataFrame({"t_start_sec": np.arange(n_sec, dtype=int), "label": sec})

# ---------- name matching (labels -> features) ----------

TOK_M = re.compile(r"(?i)\bM(\d+)\b")
TOK_D = re.compile(r"(?i)\bD(\d+)\b")
TOK_PHASE = re.compile(r"(?i)\b(pre|post)\b")

def _tokens_from_name(name: str) -> Dict[str, Optional[str]]:
    base = os.path.basename(name).lower()
    base = re.sub(r"\.(features|merged|labels)\.csv$", "", base)
    base = re.sub(r"\.csv$", "", base)
    base = re.sub(r"[_\-\.\s]+", " ", base)
    m = TOK_M.search(base); d = TOK_D.search(base); p = TOK_PHASE.search(base)
    return {
        "m": m.group(1) if m else None,
        "d": d.group(1) if d else None,
        "phase": p.group(1).lower() if p else None,
    }

def _key(tok: Dict[str, Optional[str]], include_day: bool = True) -> str:
    parts: List[str] = []
    if tok.get("phase"): parts.append(tok["phase"])
    if tok.get("m"):     parts.append(f"m{tok['m']}")
    if include_day and tok.get("d"):
        parts.append(f"d{tok['d']}")
    return "_".join(parts)

def _feature_base(path: str) -> str:
    b = os.path.basename(path)
    for suf in (".features.csv", ".merged.csv"):
        if b.endswith(suf):
            return b[: -len(suf)]
    return os.path.splitext(b)[0]

# ---------- commands ----------

@app.command("one")
def convert_one(
    labels_csv: str = typer.Option(..., help="Frame-level label CSV"),
    out_csv: str = typer.Option(..., help="Output per-second label CSV (matched or not)"),
    fps: float = typer.Option(30.0),
    min_immobility_s: float = typer.Option(1.0),
    rule: str = typer.Option("majority", help="'majority' or 'any' after de-spike"),
    trim_to_seconds: Optional[int] = typer.Option(None, help="If set, trim to this many seconds"),
):
    """Convert a single frame-level label file to per-second labels."""
    y = _read_label_series(labels_csv)
    y2 = _despike_by_duration(y, fps=fps, min_immobility_s=min_immobility_s)
    df = _downsample_to_seconds(y2, fps=fps, rule=rule)
    if trim_to_seconds is not None:
        df = df.iloc[:int(trim_to_seconds)].reset_index(drop=True)
    _ensure_dir(out_csv)
    df.to_csv(out_csv, index=False)
    print(f"[green]Wrote[/green] {out_csv}  (seconds={len(df)}, from frames={len(y)})")

@app.command("batch-map")
def convert_batch_map(
    features_glob: str = typer.Option(..., help="Glob for features or merged CSVs (per-second)"),
    labels_glob: str   = typer.Option(..., help="Glob for frame-level label CSVs"),
    out_dir: str       = typer.Option("labels", help="Where to write per-second labels named after features"),
    fps: float         = typer.Option(30.0),
    min_immobility_s: float = typer.Option(1.0),
    rule: str          = typer.Option("majority"),
):
    """
    Convert many frame-level label CSVs to per-second, map each to a feature base name,
    trim to the feature's number of seconds, and write as labels/<feature_base>.labels.csv.
    """
    f_paths = sorted(glob.glob(features_glob))
    l_paths = sorted(glob.glob(labels_glob))
    if not f_paths:
        raise typer.BadParameter(f"No features matched: {features_glob}")
    if not l_paths:
        raise typer.BadParameter(f"No label files matched: {labels_glob}")

    # Build feature index by canonical key (with and without day)
    feat_by_key: Dict[str, List[Tuple[str,int]]] = {}
    for fp in f_paths:
        base = _feature_base(fp)
        tok = _tokens_from_name(base)
        # determine seconds length from feature file
        try:
            fdf = pd.read_csv(fp, usecols=["t_start_sec"])
            n_sec = len(fdf)
        except Exception:
            fdf = pd.read_csv(fp)
            n_sec = len(fdf)  # fallback: row count
        for include_day in (True, False):
            k = _key(tok, include_day=include_day)
            if not k:
                continue
            feat_by_key.setdefault(k, []).append((base, n_sec))

    os.makedirs(out_dir, exist_ok=True)
    matched, unmatched = 0, 0

    for lp in l_paths:
        ltok = _tokens_from_name(lp)
        k_full = _key(ltok, include_day=True)
        k_rel  = _key(ltok, include_day=False)

        cands = feat_by_key.get(k_full) or feat_by_key.get(k_rel) or []
        if not cands:
            print(f"[yellow]WARN[/yellow] no feature match for label '{os.path.basename(lp)}'")
            unmatched += 1
            continue

        # If multiple candidates, prefer exact day match
        chosen_base, chosen_len = None, None
        if k_full in feat_by_key and len(feat_by_key[k_full]) == 1:
            chosen_base, chosen_len = feat_by_key[k_full][0]
        else:
            # pick first; optionally refine by day equality
            for b, n in cands:
                btok = _tokens_from_name(b)
                if ltok.get("d") and btok.get("d") == ltok.get("d"):
                    chosen_base, chosen_len = b, n
                    break
            if chosen_base is None:
                chosen_base, chosen_len = cands[0]

        # Convert + trim
        y = _read_label_series(lp)
        y2 = _despike_by_duration(y, fps=fps, min_immobility_s=min_immobility_s)
        df = _downsample_to_seconds(y2, fps=fps, rule=rule)
        df = df.iloc[:chosen_len].reset_index(drop=True)

        out_path = os.path.join(out_dir, f"{chosen_base}.labels.csv")
        df.to_csv(out_path, index=False)
        print(f"[green]OK[/green] {os.path.basename(lp)} -> {out_path}  (seconds={len(df)})")
        matched += 1

    print(f"[bold]Done[/bold]. Matched {matched}  |  Unmatched {unmatched}")
