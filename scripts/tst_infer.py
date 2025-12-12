import typer
import pandas as pd
import numpy as np
import joblib
from rich import print
from tst.postprocess import median_smooth
# add imports
import os, glob, json
from typing import List, Optional
from sklearn.ensemble import RandomForestClassifier  # type only
from tst.io import load_area_and_similarity
from tst.features import extract_hard_features, apply_rule_and_uncertainty
from tst.postprocess import median_smooth  # already imported above
# Optional backbone (only used if yolo-model is provided)
try:
    from ultralytics import YOLO
    from tst.featurizers.ultra_backbone import UltralyticsSegFeaturizer
    _HAS_YOLO = True
except Exception:
    _HAS_YOLO = False
CONTROL_COLS = {
    "t_start_sec", "label", "state_rule", "rule_pred", "is_uncertain",
}

def _ensure_dir(p: str):
    d = os.path.dirname(p) or "."
    os.makedirs(d, exist_ok=True)

def _base_no_ext(p: str):
    b = os.path.basename(p)
    return os.path.splitext(b)[0]

def _align_X(df: pd.DataFrame, model_path: str) -> pd.DataFrame:
    """Align columns to what the model was trained on if meta is present; else use all numeric minus control cols."""
    meta_path = model_path + ".meta.json"
    if os.path.isfile(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        cols = meta.get("feature_columns", [])
        X = pd.DataFrame(index=df.index)
        for c in cols:
            X[c] = df[c] if c in df.columns else 0.0
        return X
    # fallback: auto-pick numeric columns (excluding control)
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    num = [c for c in num if c not in CONTROL_COLS]
    return df[num]

def _predict_with_verifier(df: pd.DataFrame, clf, smooth_k: int = 7, only_uncertain: bool = True) -> pd.DataFrame:
    """Create verifier_proba/pred and final_pred on a copy of df."""
    out = df.copy()
    # accept either 'state_rule' or 'rule_pred'
    rule_col = "state_rule" if "state_rule" in out.columns else ("rule_pred" if "rule_pred" in out.columns else None)
    if rule_col is None and only_uncertain:
        raise RuntimeError("No rule column found ('state_rule' or 'rule_pred'). Re-run features/rule or disable only_uncertain.")
    if "is_uncertain" not in out.columns and only_uncertain:
        raise RuntimeError("Missing 'is_uncertain' column. Re-run features with threshold/band or disable only_uncertain.")

    # default = rule
    if rule_col is not None:
        state = out[rule_col].astype(int).values.copy().astype(float)
    else:
        state = np.zeros(len(out), dtype=float)

    # rows to score
    mask = (out["is_uncertain"] == 1) if (only_uncertain and "is_uncertain" in out.columns) else np.ones(len(out), bool)
    if mask.any():
        X = _align_X(out, model_path=clf.__dict__.get("model_path", "")) if hasattr(clf, "__dict__") else _align_X(out, "")
        X = X.values.astype(np.float32, copy=False)
        if hasattr(clf, "predict_proba"):
            pp = clf.predict_proba(X)
            # Handle binary/degenerate cases robustly
            classes = getattr(clf, "classes_", None)
            if classes is None:
                # Fallback: assume positive class=1 is column 1 if exists
                proba = pp[:, 1] if pp.shape[1] > 1 else np.zeros(len(X), dtype=float)
            else:
                classes = np.asarray(classes)
                if classes.size == 2:
                    # find the column that corresponds to label 1
                    pos_idx = int(np.where(classes == 1)[0][0]) if (classes == 1).any() else 1
                    proba = pp[:, pos_idx]
                elif classes.size == 1:
                    # model learned a single class; probability is constant
                    c = int(classes[0])
                    proba = np.full(len(X), 1.0 if c == 1 else 0.0, dtype=float)
                else:
                    # multi-class (unexpected here) → take max prob as a proxy
                    proba = pp.max(axis=1)
        elif hasattr(clf, "decision_function"):
            s = clf.decision_function(X)
            proba = (s - s.min()) / (s.max() - s.min() + 1e-8)
        else:
            proba = clf.predict(X).astype(float)
        proba_sm = median_smooth(proba, k=smooth_k)
        pred = (proba_sm >= 0.5).astype(int)

        out["verifier_proba"] = proba_sm
        out["verifier_pred"] = pred
        if only_uncertain and "is_uncertain" in out.columns:
            state[mask] = pred[mask]
        else:
            state = pred.astype(float)
    else:
        out["verifier_proba"] = np.nan
        out["verifier_pred"] = np.nan

    out["final_pred"] = state.astype(int)
    return out

app = typer.Typer()

@app.command()
def infer(
    features_csv: str = typer.Option("tst_features_persec.csv"),
    model_path: str = typer.Option("verifier_rf.joblib"),
    out_csv: str = typer.Option("tst_predictions.csv"),
    smooth_k: int = typer.Option(7),
):
    df = pd.read_csv(features_csv)
    clf = joblib.load(model_path)
    df_easy = df[df["is_uncertain"] == 0].copy()
    df_hard = df[df["is_uncertain"] == 1].copy()

    # Start with rule-based state
    state = df["state_rule"].values.copy().astype(float)

    # Replace uncertain with model probability>0.5
    if len(df_hard):
        probs = clf.predict_proba(df_hard)
        probs_sm = median_smooth(probs, k=smooth_k)
        preds = (probs_sm >= 0.5).astype(int)
        state[df_hard.index] = preds

    out = df.copy()
    out["state_final"] = state.astype(int)
    out.to_csv(out_csv, index=False)
    print(f"[green]Saved predictions[/green] -> {out_csv}")

@app.command("infer-batch")
def infer_batch(
    features_glob: str = typer.Option(..., help="Glob for *.features.csv or *.merged.csv"),
    model_path: str = typer.Option("verifier_rf.joblib"),
    out_dir: str = typer.Option("predictions"),
    smooth_k: int = typer.Option(7),
    only_uncertain: bool = typer.Option(True),
):
    """Apply verifier to many feature CSVs and write *.with_verifier.csv files."""
    paths = sorted(glob.glob(features_glob))
    if not paths:
        raise typer.BadParameter(f"No inputs matched: {features_glob}")
    clf = joblib.load(model_path)
    # save model_path into clf for _align_X to find meta
    try: setattr(clf, "model_path", model_path)
    except Exception: pass
    os.makedirs(out_dir, exist_ok=True)
    wrote = 0
    for p in paths:
        df = pd.read_csv(p)
        out = _predict_with_verifier(df, clf, smooth_k=smooth_k, only_uncertain=only_uncertain)
        name = _base_no_ext(p) + ".with_verifier.csv"
        out_path = os.path.join(out_dir, os.path.basename(name))
        _ensure_dir(out_path)
        out.to_csv(out_path, index=False)
        print(f"[green]OK[/green] {p} -> {out_path}")
        wrote += 1
    print(f"[bold]Done.[/bold] Wrote {wrote} files.")

@app.command("from-npy")
def from_npy(
    npy_path: str = typer.Option(..., help="Path to output_data.npy (area[,similarity])"),
    model_path: str = typer.Option("verifier_rf.joblib"),
    out_csv: str = typer.Option("predictions.csv"),
    # engineered feature params
    fps: int = typer.Option(30),
    win_s: float = typer.Option(1.0),
    threshold: float = typer.Option(0.02),
    band: float = typer.Option(0.30),
    beta: float = typer.Option(0.31),
    w_min: float = typer.Option(0.75),
    # optional backbone
    yolo_model: Optional[str] = typer.Option(None, help="Ultralytics seg model path (e.g., best.pt)"),
    frames_dir: Optional[str] = typer.Option(None, help="Folder of frames for this session"),
    layer_paths: List[str] = typer.Option([], help="Repeat per layer: --layer-paths model.18"),
    imgsz: int = typer.Option(640),
    batch_size: int = typer.Option(1),
    device: str = typer.Option("cuda"),
    fp16: bool = typer.Option(True),
    smooth_k: int = typer.Option(7),
    only_uncertain: bool = typer.Option(True),
):
    """End-to-end: NPY -> engineered features (+ optional backbone) -> rule -> verifier -> final_pred."""
    # 1) engineered features
    area, sim = load_area_and_similarity(npy_path)
    df = extract_hard_features(area, sim, fps=fps, win_s=win_s,
                               beta=beta, w_min=w_min, norm_within_video=True, add_lowfreq_power=True)
    df = apply_rule_and_uncertainty(df, threshold=threshold, band=band)

    # 2) optional backbone
    if yolo_model and frames_dir:
        if not _HAS_YOLO:
            raise RuntimeError("Ultralytics/YOLO not available; install the extras or omit --yolo-model.")
        # light, single-chunk pass: take largest mask per frame, ROI pool neck features, average per second
    import cv2, glob as _glob, torch

    # --- config
    use_cuda = (device == "cuda") and torch.cuda.is_available()
    dev = torch.device("cuda" if use_cuda else "cpu")
    use_fp16 = bool(fp16 and use_cuda)

    # load YOLO and grab the nn.Module
    yolo = YOLO(yolo_model)
    net = yolo.model.to(dev).eval()
    if use_fp16:
        net.half()

    # choose one feature layer (keep it single to save memory)
    target = net
    # default to model.18 if none provided
    sel = (layer_paths or ["model.18"])[0]
    for name in sel.split(".")[1:]:
        target = getattr(target, name)

    # collect frame paths
    frames = sorted(_glob.glob(os.path.join(frames_dir, "*.jpg")) +
                    _glob.glob(os.path.join(frames_dir, "*.png")))
    if not frames:
        raise RuntimeError(f"No frames found in {frames_dir}")

    # seconds bookkeeping
    win = int(round(win_s * fps))
    n_frames = len(frames)
    n_secs = n_frames // win

    feat_dim = None
    sec_sum = None
    sec_cnt = np.zeros(n_secs, dtype=np.int32)

    def preprocess(path, size):
        img = cv2.imread(path)  # BGR uint8
        if img is None:
            # robust to read errors
            img = np.zeros((size, size, 3), np.uint8)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
        img = img[:, :, ::-1]  # BGR->RGB
        x = img.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))  # CHW
        return x

    # hook that does GAP and CPU-accumulate per-second
    def make_hook(sec_indices_for_batch):
        def _hook(_m, _inp, out):
            with torch.no_grad():
                # out: [B, C, H, W]
                feat = out.float().mean(dim=(2, 3))  # GAP -> [B, C]
                f_cpu = feat.detach().cpu().numpy()
            nonlocal feat_dim, sec_sum
            if feat_dim is None:
                feat_dim = f_cpu.shape[1]
                sec_sum = np.zeros((n_secs, feat_dim), dtype=np.float32)
            for i, sidx in enumerate(sec_indices_for_batch):
                if 0 <= sidx < n_secs:
                    sec_sum[sidx] += f_cpu[i]
                    sec_cnt[sidx] += 1
        return _hook

    # stream frames in tiny batches
    cur_batch, cur_secs = [], []
    with torch.inference_mode():
        for f_idx, fp_ in enumerate(frames[: n_secs * win]):  # drop partial last second
            cur_batch.append(preprocess(fp_, imgsz))
            cur_secs.append(f_idx // win)

            if len(cur_batch) == batch_size:
                x = torch.from_numpy(np.stack(cur_batch))  # [B,3,H,W]
                x = x.to(dev, non_blocking=True)
                if use_fp16: x = x.half()

                h = target.register_forward_hook(make_hook(cur_secs))
                _ = net(x)
                h.remove()

                del x; cur_batch.clear(); cur_secs.clear()
                if use_cuda:
                    torch.cuda.empty_cache()

        # flush last partial batch
        if cur_batch:
            x = torch.from_numpy(np.stack(cur_batch)).to(dev, non_blocking=True)
            if use_fp16: x = x.half()
            h = target.register_forward_hook(make_hook(cur_secs))
            _ = net(x)
            h.remove()
            del x; cur_batch.clear(); cur_secs.clear()
            if use_cuda:
                torch.cuda.empty_cache()

    # finalize per-second features
    sec_cnt[sec_cnt == 0] = 1
    bb = sec_sum / sec_cnt[:, None]  # [n_secs, C]
    bb_cols = [f"bb_{i:04d}" for i in range(bb.shape[1])]
    bb_df = pd.DataFrame(bb, columns=bb_cols)
    bb_df.insert(0, "t_start_sec", df["t_start_sec"][:len(bb_df)].values)
    # keep timeline consistent: trim df to matched seconds, then merge
    df = df.iloc[:len(bb_df)].merge(bb_df, on="t_start_sec", how="left")

    # 3) verifier
    clf = joblib.load(model_path)
    try: setattr(clf, "model_path", model_path)
    except Exception: pass
    out = _predict_with_verifier(df, clf, smooth_k=smooth_k, only_uncertain=only_uncertain)
    _ensure_dir(out_csv); out.to_csv(out_csv, index=False)
    print(f"[green]Saved predictions[/green] -> {out_csv}")
@app.command("from-npy-batch")
def from_npy_batch(
    npy_glob: str = typer.Option(..., help="Glob for *.npy (area[,similarity])"),
    model_path: str = typer.Option("verifier_rf.joblib"),
    out_dir: str = typer.Option("predictions"),
    fps: int = typer.Option(30),
    win_s: float = typer.Option(1.0),
    threshold: float = typer.Option(0.02),
    band: float = typer.Option(0.30),
    beta: float = typer.Option(0.31),
    w_min: float = typer.Option(0.75),
    frames_root: Optional[str] = typer.Option(None, help="Root folder containing <base>_images/ if adding backbone"),
    yolo_model: Optional[str] = typer.Option(None, help="Ultralytics seg model"),
    layer_paths: List[str] = typer.Option([], help="Repeat per layer hook"),
    imgsz: int = typer.Option(640),
    smooth_k: int = typer.Option(7),
    only_uncertain: bool = typer.Option(True),
):
    paths = sorted(glob.glob(npy_glob))
    if not paths:
        raise typer.BadParameter(f"No NPY matched: {npy_glob}")
    os.makedirs(out_dir, exist_ok=True)
    for p in paths:
        base = _base_no_ext(p)
        out_csv = os.path.join(out_dir, f"{base}.with_verifier.csv")
        frames_dir = None
        if frames_root:
            # support <frames_root>/<base>_images
            cand = os.path.join(frames_root, f"{base}_images")
            if os.path.isdir(cand): frames_dir = cand
        # reuse the single-file logic
        from_npy(  # call command function as regular function
            npy_path=p, model_path=model_path, out_csv=out_csv,
            fps=fps, win_s=win_s, threshold=threshold, band=band, beta=beta, w_min=w_min,
            yolo_model=yolo_model, frames_dir=frames_dir, layer_paths=layer_paths, imgsz=imgsz,
            smooth_k=smooth_k, only_uncertain=only_uncertain
        )

if __name__ == "__main__":
    app()
