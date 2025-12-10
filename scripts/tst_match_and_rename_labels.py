import re, os, glob, shutil, pandas as pd
import typer
from rich import print

app = typer.Typer()

TOK_M = re.compile(r"(?i)\bM(\d+)\b")
TOK_D = re.compile(r"(?i)\bD(\d+)\b")
TOK_PHASE = re.compile(r"(?i)\b(pre|post)\b")

def tokens_from_name(name: str):
    # lower, strip suffixes
    base = os.path.basename(name).lower()
    base = base.replace(".features.csv","").replace(".labels.csv","").replace(".csv","")
    # turn separators to spaces
    base = re.sub(r"[_\-\.\s]+", " ", base)
    m = TOK_M.search(base)
    d = TOK_D.search(base)
    p = TOK_PHASE.search(base)
    return {
        "m": m.group(1) if m else None,
        "d": d.group(1) if d else None,
        "phase": p.group(1).lower() if p else None,
    }

def canonical_key(tok: dict):
    # key ignores prefixes like "khc2" or "tst_manual_scoring"
    # include only detected tokens; missing 'd' is allowed
    parts = []
    if tok["phase"]: parts.append(tok["phase"])   # pre/post first
    if tok["m"]:     parts.append(f"m{tok['m']}")
    if tok["d"]:     parts.append(f"d{tok['d']}")
    return "_".join(parts)

def feature_base_from_path(path: str):
    b = os.path.basename(path)
    b = b.replace(".features.csv","").replace(".merged.csv","")
    return b

@app.command()
def run(
    features_glob: str = typer.Option("data/features/*.features.csv"),
    labels_glob: str   = typer.Option("processed_labels/*.labels.csv"),
    out_map_csv: str   = typer.Option("label_feature_mapping.csv"),
    out_labels_dir: str= typer.Option("labels", help="Where to write renamed label files"),
    copy_files: bool   = typer.Option(True, help="Copy labels to out_labels_dir with feature base names"),
):
    feat_paths = sorted(glob.glob(features_glob))
    lab_paths  = sorted(glob.glob(labels_glob))
    if not feat_paths: raise typer.BadParameter(f"No features matched: {features_glob}")
    if not lab_paths:  raise typer.BadParameter(f"No labels matched: {labels_glob}")

    # index features by canonical key
    feat_index = {}
    feat_rows = []
    for fp in feat_paths:
        base = feature_base_from_path(fp)
        tok = tokens_from_name(base)
        key = canonical_key(tok)
        feat_index.setdefault(key, []).append(base)
        feat_rows.append({"feature_base": base, "key": key, **tok})

    # find best label→feature match
    rows = []
    unmatched = []
    for lp in lab_paths:
        lbase = os.path.basename(lp)
        ltok  = tokens_from_name(lbase)
        lkey  = canonical_key(ltok)
        candidates = feat_index.get(lkey, [])
        if not candidates:
            # try relaxed key (ignore day if missing on either side)
            lkey_relaxed = "_".join([p for p in [ltok["phase"], f"m{ltok['m']}" if ltok["m"] else None] if p])
            cand_relaxed = [fb for fb,t in [(feature_base_from_path(fp), tokens_from_name(feature_base_from_path(fp))) for fp in feat_paths]
                            if canonical_key({"phase":t["phase"],"m":t["m"],"d":None}) == lkey_relaxed]
            if cand_relaxed:
                candidates = cand_relaxed
        if candidates:
            # if multiple (rare), pick the one that shares 'd' if present
            chosen = None
            if ltok["d"]:
                for c in candidates:
                    if tokens_from_name(c)["d"] == ltok["d"]:
                        chosen = c; break
            if chosen is None:
                chosen = candidates[0]
            rows.append({"label_file": lp, "feature_base": chosen, "key": lkey, **ltok})
        else:
            unmatched.append(lp)

    df_map = pd.DataFrame(rows)
    df_feat = pd.DataFrame(feat_rows)
    df_map.to_csv(out_map_csv, index=False)
    print(f"[green]Wrote mapping[/green] -> {out_map_csv} (matched {len(df_map)} / labels={len(lab_paths)})")
    if unmatched:
        print(f"[yellow]Unmatched labels ({len(unmatched)}):[/yellow]")
        for u in unmatched[:10]:
            print("  -", os.path.basename(u))
        if len(unmatched) > 10:
            print("  ...")

    if copy_files:
        os.makedirs(out_labels_dir, exist_ok=True)
        for _, r in df_map.iterrows():
            dst = os.path.join(out_labels_dir, f"{r['feature_base']}.labels.csv")
            shutil.copy2(r["label_file"], dst)
        print(f"[green]Copied[/green] {len(df_map)} labels to: {out_labels_dir}")

if __name__ == "__main__":
    app()
