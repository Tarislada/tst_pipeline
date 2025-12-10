# scripts/tst_check_alignment.py
import os, glob
import pandas as pd
import typer
from rich import print
from typing import Optional

app = typer.Typer(help="Check alignment between per-second features and labels.")

def _basename_no_suffix(path: str) -> str:
    b = os.path.basename(path)
    if b.endswith(".features.csv"):
        return b[: -len(".features.csv")]
    if b.endswith(".merged.csv"):
        return b[: -len(".merged.csv")]
    if b.endswith(".labels.csv"):
        return b[: -len(".labels.csv")]
    return os.path.splitext(b)[0]

@app.command("run")
def run_single(
    features_csv: str = typer.Option(..., help="CSV with per-second features (or merged features)"),
    labels_csv: str = typer.Option(..., help="CSV with per-second labels"),
):
    f = pd.read_csv(features_csv)
    l = pd.read_csv(labels_csv)
    m = f.merge(l, on="t_start_sec", how="left", suffixes=("", "_lab"))
    missing = int(m["label"].isna().sum())
    print(f"[cyan]Feature rows[/cyan]: {len(f)}  |  [cyan]Labels[/cyan]: {len(l)}  |  "
          f"[yellow]Missing labels after merge[/yellow]: {missing}")
    if missing:
        ex = m.loc[m["label"].isna(), ["t_start_sec"]].head(10)
        print("[red]Examples with missing labels:[/red]")
        print(ex.to_string(index=False))
    else:
        print("[green]All feature windows have labels.[/green]")

@app.command("run-batch")
def run_batch(
    features_glob: str = typer.Option("data/features/*.features.csv", help="Glob for features or merged CSVs"),
    labels_glob: str = typer.Option("labels/*.labels.csv", help="Glob for labels"),
    out_report: Optional[str] = typer.Option("alignment_report.csv", help="Where to save summary CSV"),
    show_examples: int = typer.Option(3, help="Missing-time examples to print per session"),
):
    f_paths = sorted(glob.glob(features_glob))
    l_paths = sorted(glob.glob(labels_glob))
    if not f_paths:
        raise typer.BadParameter(f"No features matched: {features_glob}")
    if not l_paths:
        raise typer.BadParameter(f"No labels matched: {labels_glob}")

    label_map = {_basename_no_suffix(p): p for p in l_paths}
    rows = []
    for fp in f_paths:
        base = _basename_no_suffix(fp)
        lp = label_map.get(base)
        if lp is None:
            print(f"[yellow]WARN[/yellow] No labels for base '{base}' (features={fp})")
            rows.append({
                "base": base, "features_path": fp, "labels_path": None,
                "n_features": None, "n_labels": None,
                "missing_labels": None, "covered": 0.0
            })
            continue

        f = pd.read_csv(fp)
        l = pd.read_csv(lp)
        m = f.merge(l, on="t_start_sec", how="left", suffixes=("", "_lab"))
        missing = int(m["label"].isna().sum())
        covered = 1.0 - (missing / max(len(f), 1))

        rows.append({
            "base": base,
            "features_path": fp,
            "labels_path": lp,
            "n_features": len(f),
            "n_labels": len(l),
            "missing_labels": missing,
            "covered": covered
        })

        if missing > 0 and show_examples > 0:
            ex = m.loc[m["label"].isna(), ["t_start_sec"]].head(show_examples)
            if not ex.empty:
                print(f"[red]MISSING[/red] {base}: first {len(ex)} missing seconds:")
                print(ex.to_string(index=False))

    report = pd.DataFrame(rows).sort_values(["covered", "base"], ascending=[True, True])
    if out_report:
        report.to_csv(out_report, index=False)
        print(f"[green]Wrote report[/green] -> {out_report}")
    print(report.to_string(index=False))

if __name__ == "__main__":
    app()
