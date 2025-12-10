import typer, numpy as np, pandas as pd
from rich import print
from pathlib import Path
from typing import List, Optional
from ultralytics import YOLO
from tst.featurizers.ultra_backbone import UltralyticsSegFeaturizer
from tst.io import load_area_and_similarity
from tst.features import extract_hard_features, apply_rule_and_uncertainty
import math, gc, torch
import cv2

app = typer.Typer()

@app.command()
def run(
    model_path: str = typer.Option(..., help="Ultralytics YOLO-seg model path (e.g., best.pt)."),
    frames_dir: str = typer.Option(..., help="Directory of frames (000001.jpg ...)"),
    layer_paths: List[str] = typer.Option([], help="Dotted paths inside model.model to hook, e.g. 'model.22.cv3'."),
    imgsz: int = typer.Option(640),
    fps: int = typer.Option(30),
    win_s: float = typer.Option(1.0),
    input_npy: str = typer.Option("output_data.npy"),
    out_features_csv: str = typer.Option("tst_features_persec.csv"),
    out_bb_csv: str = typer.Option("tst_backbone_embeds.csv"),
    out_merged_csv: str = typer.Option("tst_features_plus_backbone.csv"),
):
    # 1) Hard features
    area, sim = load_area_and_similarity(input_npy)
    df = extract_hard_features(area, sim, fps=fps, win_s=win_s, beta=0.31, w_min=0.75, norm_within_video=True, add_lowfreq_power=True)
    df = apply_rule_and_uncertainty(df, threshold=0.02, band=0.30)
    df.to_csv(out_features_csv, index=False)
    print(f"[green]Saved hard features[/green] -> {out_features_csv}")

    # 2) Prepare frames
    frame_files = sorted([p for p in Path(frames_dir).glob("*.jpg")] + [p for p in Path(frames_dir).glob("*.png")])
    assert len(frame_files) > 0, "No frames found"
    frames_bgr = [cv2.imread(str(f)) for f in frame_files]

    # 3) Run YOLO once for masks/boxes (choose largest instance per frame)
    yolo = YOLO(model_path)
    results = yolo.predict(frames_bgr, imgsz=imgsz, verbose=False, stream=True)
    boxes_per_frame, masks_per_frame = [], []
    for r in results:
        H, W = r.orig_img.shape[:2]
        if (r.masks is None) or (r.masks.data is None) or (len(r.boxes) == 0):
            boxes_per_frame.append(np.zeros((0,4), dtype=np.float32))
            masks_per_frame.append(np.zeros((0, H, W), dtype=np.uint8))
            continue
        mdata = r.masks.data.cpu().numpy()
        up = []
        areas = []
        for m in mdata:
            m_img = cv2.resize((m*255).astype("uint8"), (W, H), interpolation=cv2.INTER_NEAREST)
            binm = (m_img > 127).astype("uint8")
            up.append(binm)
            areas.append(np.sum(binm))
        idx = int(np.argmax(areas)) if areas else -1
        if idx < 0:
            boxes_per_frame.append(np.zeros((0,4), dtype=np.float32))
            masks_per_frame.append(np.zeros((0, H, W), dtype=np.uint8))
            continue
        mask = up[idx]
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            boxes_per_frame.append(np.zeros((0,4), dtype=np.float32))
            masks_per_frame.append(np.zeros((0, H, W), dtype=np.uint8))
            continue
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        boxes_per_frame.append(np.array([[x1,y1,x2,y2]], dtype=np.float32))
        masks_per_frame.append(np.expand_dims(mask, 0).astype(np.uint8))

    # 4) Hook layers and extract backbone embeddings
    feat = UltralyticsSegFeaturizer(yolo, layer_paths=layer_paths, imgsz=imgsz, device="cuda")
    embeds = feat.extract_batch(frames_bgr, boxes_per_frame, masks_per_frame)  # (T, Ctot)

    # 5) Pool per-second (simple mean)
    T = len(frames_bgr)
    win = int(round(win_s * fps))
    pooled = []
    for s in range(0, max(0, T - win + 1), win):
        chunk = embeds[s:s+win]
        if len(chunk) == 0:
            pooled.append(None)
        else:
            pooled.append(chunk.mean(axis=0))
    pooled = pooled[:len(df)] + [None]*max(0, len(df)-len(pooled))
    bb_df = pd.DataFrame({"t_start_sec": df["t_start_sec"]})
    if pooled[0] is not None:
        C = len(pooled[0])
        for c in range(C):
            bb_df[f"bb_{c:04d}"] = [row[c] if row is not None else np.nan for row in pooled]
    bb_df.to_csv(out_bb_csv, index=False)
    print(f"[green]Saved backbone per-second[/green] -> {out_bb_csv}")

    merged = df.merge(bb_df, on="t_start_sec", how="left")
    merged.to_csv(out_merged_csv, index=False)
    print(f"[green]Saved merged features[/green] -> {out_merged_csv}")

@app.command(name="run-batch")
def run_batch(
    model_path: str = typer.Option(..., help="Ultralytics YOLO-seg model (e.g., best.pt)"),
    frames_root: str = typer.Option(..., help="Root with per-session frame folders (e.g., data/processed_images)"),
    layer_paths: List[str] = typer.Option([], help="Repeat per layer: --layer-paths model.12 --layer-paths model.15"),
    imgsz: int = typer.Option(640),
    fps: int = typer.Option(30),
    win_s: float = typer.Option(1.0),
    npy_root: Optional[str] = typer.Option(None, help="Where per-session .npy files live (e.g., data/processed_videos). If None, also try inside each frames folder."),
    out_features_dir: str = typer.Option("features"),
    out_bb_dir: str = typer.Option("backbone"),
    out_merged_dir: str = typer.Option("merged"),
    reuse_features_dir: Optional[str] = typer.Option(None, help="If set, reuse existing features here (e.g., data/features). Else compute from NPY."),
    threshold: float = typer.Option(0.02, help="Rule threshold"),
    band: float = typer.Option(0.30, help="Uncertain band width"),
    beta: float = typer.Option(0.31),
    w_min: float = typer.Option(0.75),
    chunk_frames: int = typer.Option(256, help="Process this many frames per chunk"),
    yolo_batch: int = typer.Option(16, help="YOLO predict batch size per chunk"),
):
    """
    Batch process sessions: for each frames subfolder under frames_root,
      1) (re)use or compute engineered features from the session's output_data.npy,
      2) stream YOLO segmentation + backbone hooks in chunks,
      3) pool per-second on the fly and merge.

    Assumptions that match your layout:
      - Frames live in:  data/processed_images/<session>_images/0.jpg,1.jpg,...
      - Session base is derived by stripping suffixes: _images, _frames, _imgs
      - NPYs live in:    data/processed_videos/<base>.npy   (or <base>_data.npy, <base>_post_data.npy)
      - Outputs go to:   out_features_dir/<base>.features.csv, out_bb_dir/<base>.bb.csv, out_merged_dir/<base>.merged.csv
    """
    import os, glob, math, gc
    import numpy as np
    import pandas as pd
    import cv2
    import torch
    from rich import print
    from ultralytics import YOLO
    from tst.io import load_area_and_similarity
    from tst.features import extract_hard_features, apply_rule_and_uncertainty
    from tst.featurizers.ultra_backbone import UltralyticsSegFeaturizer

    os.makedirs(out_features_dir, exist_ok=True)
    os.makedirs(out_bb_dir, exist_ok=True)
    os.makedirs(out_merged_dir, exist_ok=True)

    def base_from_frames_dir(dname: str) -> str:
        base = os.path.basename(dname.rstrip("/"))
        for suf in ("_images", "_frames", "_imgs"):
            if base.endswith(suf):
                base = base[: -len(suf)]
                break
        return base

    def find_npy(base: str, session_dir: str) -> Optional[str]:
        # try inside frames folder first (some users keep it there)
        inside = os.path.join(session_dir, "output_data.npy")
        if os.path.isfile(inside):
            return inside
        # then look in npy_root if provided
        if npy_root:
            candidates = [
                os.path.join(npy_root, f"{base}.npy"),
                os.path.join(npy_root, f"{base}_data.npy"),
                os.path.join(npy_root, f"{base}_post_data.npy"),
            ]
            for c in candidates:
                if os.path.isfile(c):
                    return c
        return None

    # discover session frame dirs (immediate subdirectories)
    sessions = [p for p in sorted(glob.glob(os.path.join(frames_root, "*"))) if os.path.isdir(p)]
    if not sessions:
        raise typer.BadParameter(f"No session folders found under: {frames_root}")

    # init model/featurizer once
    yolo = YOLO(model_path)
    feat = UltralyticsSegFeaturizer(yolo, layer_paths=layer_paths, imgsz=imgsz, device="cuda")

    ok = fail = 0
    for sess_dir in sessions:
        base = base_from_frames_dir(sess_dir)

        # frame list
        frame_files = sorted(
            glob.glob(os.path.join(sess_dir, "*.jpg")) +
            glob.glob(os.path.join(sess_dir, "*.png"))
        )
        if not frame_files:
            print(f"[yellow]SKIP[/yellow] {base}: no frames in {sess_dir}")
            fail += 1
            continue

        # npy
        npy_path = find_npy(base, sess_dir)
        if npy_path is None:
            print(f"[yellow]SKIP[/yellow] {base}: no NPY found (checked inside session and in {npy_root})")
            fail += 1
            continue

        try:
            # (1) engineered features: reuse or compute
            f_out = os.path.join(out_features_dir, f"{base}.features.csv")
            if reuse_features_dir:
                reuse_path = os.path.join(reuse_features_dir, f"{base}.features.csv")
                if os.path.isfile(reuse_path):
                    df = pd.read_csv(reuse_path)
                else:
                    area, sim = load_area_and_similarity(npy_path)
                    df = extract_hard_features(area, sim, fps=fps, win_s=win_s,
                                               beta=beta, w_min=w_min,
                                               norm_within_video=True, add_lowfreq_power=True)
                    df = apply_rule_and_uncertainty(df, threshold=threshold, band=band)
                    df.to_csv(f_out, index=False)
            else:
                area, sim = load_area_and_similarity(npy_path)
                df = extract_hard_features(area, sim, fps=fps, win_s=win_s,
                                           beta=beta, w_min=w_min,
                                           norm_within_video=True, add_lowfreq_power=True)
                df = apply_rule_and_uncertainty(df, threshold=threshold, band=band)
                df.to_csv(f_out, index=False)

            # (2) streaming seg + backbone in chunks, pool per-second on the fly
            T = len(frame_files)
            win_frames = int(round(win_s * fps))
            n_sec = max(0, T // win_frames)
            bb_accum = None   # (n_sec, C)
            bb_counts = None  # (n_sec,)
            feat_dim = None

            def pool_chunk_to_seconds(start_idx: int, embeds_chunk: np.ndarray):
                nonlocal bb_accum, bb_counts, feat_dim
                if embeds_chunk.size == 0:
                    return
                if feat_dim is None:
                    feat_dim = embeds_chunk.shape[1]
                    bb_accum = np.zeros((n_sec, feat_dim), dtype=np.float32)
                    bb_counts = np.zeros((n_sec,), dtype=np.int32)
                # add to per-second accumulators
                for i in range(embeds_chunk.shape[0]):
                    fidx = start_idx + i
                    sec = fidx // win_frames
                    if sec < n_sec:
                        bb_accum[sec] += embeds_chunk[i]
                        bb_counts[sec] += 1

            for s in range(0, T, chunk_frames):
                sub_files = frame_files[s : s + chunk_frames]
                frames_bgr = [cv2.imread(f) for f in sub_files]

                # 2a) segmentation for this chunk → largest mask per frame
                results = yolo.predict(
                    frames_bgr, imgsz=imgsz, verbose=False, stream=True,
                    batch=min(yolo_batch, len(frames_bgr))
                )
                boxes_per_frame, masks_per_frame = [], []
                for r in results:
                    H, W = r.orig_img.shape[:2]
                    if (r.masks is None) or (r.masks.data is None) or (len(r.boxes) == 0):
                        boxes_per_frame.append(np.zeros((0,4), dtype=np.float32))
                        masks_per_frame.append(np.zeros((0, H, W), dtype=np.uint8))
                        continue
                    mdata = r.masks.data.cpu().numpy()
                    best, best_area = None, -1
                    for m in mdata:
                        m_img = cv2.resize((m*255).astype("uint8"), (W, H), interpolation=cv2.INTER_NEAREST)
                        binm = (m_img > 127).astype("uint8")
                        a = int(binm.sum())
                        if a > best_area:
                            best_area, best = a, binm
                    if best is None:
                        boxes_per_frame.append(np.zeros((0,4), dtype=np.float32))
                        masks_per_frame.append(np.zeros((0, H, W), dtype=np.uint8))
                        continue
                    ys, xs = np.where(best > 0)
                    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
                    boxes_per_frame.append(np.array([[x1,y1,x2,y2]], dtype=np.float32))
                    masks_per_frame.append(np.expand_dims(best, 0).astype(np.uint8))

                # 2b) backbone features for this chunk only
                embeds_chunk = feat.extract_batch(frames_bgr, boxes_per_frame, masks_per_frame)  # (Nchunk, C)

                # 2c) pool to seconds & free memory
                pool_chunk_to_seconds(s, embeds_chunk)
                del frames_bgr, boxes_per_frame, masks_per_frame, embeds_chunk
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            # build backbone df aligned to engineered features
            bb_df = pd.DataFrame({"t_start_sec": df["t_start_sec"][:n_sec].to_numpy(copy=False)})
            if (bb_accum is not None) and (bb_counts is not None) and (bb_counts.sum() > 0):
                bb_mean = (bb_accum / np.maximum(bb_counts[:, None], 1)).astype(np.float32, copy=False)
                bb_cols = [f"bb_{i:04d}" for i in range(bb_mean.shape[1])]
                bb_block = pd.DataFrame(bb_mean, columns=bb_cols)
                bb_df = pd.concat([bb_df, bb_block], axis=1, copy=False)

            b_out = os.path.join(out_bb_dir, f"{base}.bb.csv")
            bb_df.to_csv(b_out, index=False)

            merged = df.iloc[:n_sec].merge(bb_df, on="t_start_sec", how="left")
            m_out = os.path.join(out_merged_dir, f"{base}.merged.csv")
            merged.to_csv(m_out, index=False)

            print(f"[green]OK[/green] {base} → {f_out} | {b_out} | {m_out}")
            ok += 1

        except Exception as e:
            print(f"[red]FAIL[/red] {base}: {e}")
            fail += 1

    print(f"[bold]Done.[/bold] Converted {ok}, failed {fail}.")

if __name__ == "__main__":
    app()
