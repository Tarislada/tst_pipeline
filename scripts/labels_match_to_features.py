# scripts/labels_match_to_features.py
import os, re, glob
import pandas as pd
import typer
from rich import print

app = typer.Typer(help="Match per-second label CSVs to feature basenames and trim to feature length.")

TOK_M = re.compile(r"(?i)\bM(\d+)\b")
TOK_D = re.compile(r"(?i)\bD(\d+)\b")
TOK_P = re.compile(r"(?i)\b(pre|post)\b")

def tokens(name: str):
    base = os.path.basename(name).lower()
    base = base.replace(".persec.labels.csv","").replace(".labels.csv","").replace(".csv","")
    base = re.sub(r"[_\-\.\s]+", " ", base)
    m = TOK_M.search(base); d = TOK_D.search(base); p = TOK_P.search(base)
    return {"m": m.group(1) if m else None,
            "d": d.group(1) if d else None,
            "p": p.group(1).lower() if p else None}

def key(t: dict):
    # canonical matching key; ignore project-specific prefixes
    parts = []
    if t["p"]: parts.append(t["p"])
    if t["m"]: parts.append(f"m{t['m']}")
    if t["d"]: parts.append(f"d{t['d']}")
    return "_".join(parts)

def base_no_suffix(path: str):
    b = os.path.basename(path)
    for suf in (".features.csv", ".merged.csv"):
        if b.endswith(suf): return b[:-len(suf)]
    return os.path.splitext(b)[0]

@app.command()
def run(
    features_glob: str = typer.Option(..., help="e.g. data/merged/*.merged.csv or data/features/*.features.csv"),
    labels_glob: str   = typer.Option(..., help="e.g. labels_sec/*.persec.labels.csv"),
    out_dir: str       = typer.Option("labels", help="Output dir for renamed, trimmed labels"),
):
    f_paths = sorted(glob.glob(features_glob))
    l_paths = sorted(glob.glob(labels_glob))
    if not f_paths: raise typer.BadParameter(f"No features matched: {features_glob}")
    if not l_paths: raise typer.BadParameter(f"No labels matched: {labels_glob}")
    os.makedirs(out_dir, exist_ok=True)

    # index features by canonical key
    feat_idx = {}
    for fp in f_paths:
        t = tokens(fp)
        feat_idx.setdefault(key(t), []).append(fp)

    matched, unmatched = 0, []
    for lp in l_paths:
        lt = tokens(lp); lk = key(lt)
        cand = feat_idx.get(lk, [])
        if not cand:
            # relaxed: ignore day if missing on either side
            lk_relaxed = "_".join([p for p in [lt["p"], f"m{lt['m']}" if lt["m"] else None] if p])
            cand = [fp for fp in f_paths if key({"p": tokens(fp)["p"], "m": tokens(fp)["m"], "d": None}) == lk_relaxed]
        if not cand:
            unmatched.append(os.path.basename(lp)); continue

        # prefer exact day match if available
        chosen = None
        if lt["d"]:
            for fp in cand:
                if tokens(fp)["d"] == lt["d"]:
                    chosen = fp; break
        if chosen is None: chosen = cand[0]

        # read lengths and trim
        f_df = pd.read_csv(chosen, nrows=1)  # fast header read to detect columns
        f_df = pd.read_csv(chosen)           # need full to count rows
        n_sec = len(f_df)
        l_df = pd.read_csv(lp)
        if "t_start_sec" not in l_df.columns or "label" not in l_df.columns:
            raise RuntimeError(f"{lp} must have columns ['t_start_sec','label'] (from labels_resample.py).")
        # ensure consecutive seconds from 0
        l_df = l_df.sort_values("t_start_sec").reset_index(drop=True)
        # trim/clip to feature length
        l_df = l_df.iloc[:n_sec].copy()
        # normalize seconds to 0..n_sec-1
        l_df["t_start_sec"] = range(len(l_df))

        out_name = base_no_suffix(chosen) + ".labels.csv"
        out_path = os.path.join(out_dir, out_name)
        l_df.to_csv(out_path, index=False)
        matched += 1
        print(f"[green]OK[/green] {os.path.basename(lp)} → {out_name} (rows={len(l_df)})")

    if unmatched:
        print(f"[yellow]Unmatched labels ({len(unmatched)}):[/yellow]")
        for u in unmatched[:10]: print("  -", u)
    print(f"[bold]Done[/bold]. matched={matched}, unmatched={len(unmatched)}")
    
if __name__ == "__main__":
    app()
