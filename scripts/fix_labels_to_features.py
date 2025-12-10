# scripts/fix_labels_to_features.py  (single-command)
import os, glob, re
import pandas as pd
import numpy as np
import typer
from rich import print

# ---- helpers to make bases line up -------------------------------------------------
def base_from_features(p: str) -> str:
    b = os.path.basename(p)
    for suf in (".merged.csv", ".features.csv"):
        if b.endswith(suf):
            return b[:-len(suf)]
    return os.path.splitext(b)[0]

def base_from_labels(p: str) -> str:
    b = os.path.basename(p)
    for suf in (
        ".labels.persec.labels.csv",  # your current case
        ".persec.labels.csv",
        ".labels.csv",
        ".csv",
    ):
        if b.endswith(suf):
            return b[:-len(suf)]
    return os.path.splitext(b)[0]

# ---- main -------------------------------------------------------------------------
def main(
    features_glob: str = typer.Option(..., help="e.g. data/merged/*.merged.csv or data/features/*.features.csv"),
    labels_glob: str   = typer.Option(..., help="e.g. labels/*.persec.labels.csv or *.labels.csv"),
    out_labels_dir: str = typer.Option("labels", help="Where to write aligned labels named after feature base"),
    trim_features: bool = typer.Option(True, help="Also trim features to label length"),
    write_features_inplace: bool = typer.Option(False, help="Overwrite feature CSV in-place (keeps .bak)"),
):
    os.makedirs(out_labels_dir, exist_ok=True)

    f_paths = sorted(glob.glob(features_glob))
    l_paths = sorted(glob.glob(labels_glob))
    if not f_paths:
        raise typer.BadParameter(f"No features matched: {features_glob}")
    if not l_paths:
        raise typer.BadParameter(f"No labels matched: {labels_glob}")

    labels_map = {base_from_labels(p): p for p in l_paths}

    matched = 0
    for fpath in f_paths:
        base = base_from_features(fpath)
        lpath = labels_map.get(base)
        if not lpath:
            print(f"[yellow]SKIP[/yellow] no labels for {base}")
            continue

        f = pd.read_csv(fpath).reset_index(drop=True)
        l = pd.read_csv(lpath).reset_index(drop=True)
        if "label" not in l.columns:
            raise RuntimeError(f"{lpath} missing 'label'")

        n = min(len(f), len(l))
        if n == 0:
            print(f"[yellow]SKIP[/yellow] {base} empty after trim")
            continue

        if len(f) != n or len(l) != n:
            print(f"[cyan]{base}[/cyan] trim to n={n} (features={len(f)}, labels={len(l)})")

        f_trim = f.iloc[:n].copy().reset_index(drop=True)
        l_trim = l.iloc[:n].copy().reset_index(drop=True)

        # normalize exact 0..n-1 timeline on both sides
        f_trim["t_start_sec"] = np.arange(n, dtype=int)
        if "t_start_sec" in l_trim.columns:
            l_trim["t_start_sec"] = np.arange(n, dtype=int)
        else:
            l_trim.insert(0, "t_start_sec", np.arange(n, dtype=int))

        # write aligned labels named after feature base
        out_labels = os.path.join(out_labels_dir, f"{base}.labels.csv")
        l_trim.to_csv(out_labels, index=False)

        if trim_features:
            if write_features_inplace:
                # backup, then overwrite original features
                bak = fpath + ".bak"
                if not os.path.exists(bak):
                    os.replace(fpath, bak)
                f_trim.to_csv(fpath, index=False)
                out_feat = fpath
            else:
                out_feat = os.path.join(os.path.dirname(fpath), f"{base}.trimmed.csv")
                f_trim.to_csv(out_feat, index=False)
        else:
            out_feat = None

        matched += 1
        msg = f"[green]OK[/green] {base}: labels->{out_labels}"
        if out_feat: msg += f", features->{out_feat}"
        print(msg)

    print(f"[bold]Done[/bold]. aligned pairs: {matched}")

if __name__ == "__main__":
    typer.run(main)
