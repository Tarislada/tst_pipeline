
import typer
import pandas as pd
import numpy as np
from rich import print
from typing import Optional

app = typer.Typer()

@app.command()
def from_bouts(
    xlsx_path: str = typer.Option(..., help="Excel with columns: start_sec, end_sec, label (0 immobile, 1 mobile)"),
    fps: int = typer.Option(30, help="Video FPS, used to translate to seconds if needed."),
    out_csv: str = typer.Option("labels.csv", help="Output CSV with per-second labels."),
    round_seconds: bool = typer.Option(True, help="Round to nearest second for t_start_sec."),
):
    df = pd.read_excel(xlsx_path)
    cols = {c.lower(): c for c in df.columns}
    if ("start_sec" in cols and "end_sec" in cols) or ("start_second" in cols and "end_second" in cols):
        s_col = cols.get("start_sec", cols.get("start_second"))
        e_col = cols.get("end_sec", cols.get("end_second"))
        t_start = df[s_col].astype(float).values
        t_end   = df[e_col].astype(float).values
    elif ("start_frame" in cols and "end_frame" in cols):
        sf, ef = cols["start_frame"], cols["end_frame"]
        t_start = df[sf].astype(float).values / fps
        t_end   = df[ef].astype(float).values / fps
    else:
        raise ValueError("Excel must have (start_sec,end_sec) or (start_frame,end_frame) columns.")
    if "label" not in cols:
        lab_col = cols.get("state", cols.get("immobile", None))
        if lab_col is None:
            raise ValueError("Excel must include a 'label' (0/1) or 'state' column.")
        labels = df[lab_col].astype(int).values
    else:
        labels = df[cols["label"]].astype(int).values
    rows = []
    for s, e, y in zip(t_start, t_end, labels):
        if e < s: s, e = e, s
        i0 = int(np.floor(s))
        i1 = int(np.ceil(e))
        for sec in range(i0, i1):
            rows.append((float(sec), int(y)))
    out = pd.DataFrame(rows, columns=["t_start_sec", "label"]).drop_duplicates(subset=["t_start_sec"], keep="last")
    out = out.sort_values("t_start_sec").reset_index(drop=True)
    out.to_csv(out_csv, index=False)
    print(f"[green]Wrote per-second labels[/green] -> {out_csv} (rows={len(out)})")

@app.command(name="from-persecond-sheet")
def from_persecond_sheet(
    xlsx_path: str = typer.Option(..., help="Excel file path."),
    sheet: Optional[str] = typer.Option(None, help="Sheet name (e.g., 'Mobility_Status')."),
    out_csv: str = typer.Option("labels.csv", help="Output CSV."),
    time_col: Optional[str] = typer.Option(None, help="Name of the time column (e.g., 'Second')."),
    label_col: Optional[str] = typer.Option(None, help="Name of the label column (e.g., 'Mobility Status')."),
):
    import pandas as pd
    df = pd.read_excel(xlsx_path, sheet_name=sheet)
    # normalize headers (lowercase, strip)
    norm_map = {c: str(c).strip().lower() for c in df.columns}
    df = df.rename(columns=norm_map)

    # resolve time column
    if time_col:
        tkey = str(time_col).strip().lower()
    else:
        for cand in ["t_start_sec","sec","second","time","t","t_sec","t (s)","s"]:
            if cand in df.columns:
                tkey = cand; break
        else:
            # fall back to row index if nothing found
            df["t_start_sec"] = range(len(df))
            tkey = "t_start_sec"

    if tkey != "t_start_sec":
        df = df.rename(columns={tkey: "t_start_sec"})

    # resolve label column
    if label_col:
        lkey = str(label_col).strip().lower()
    else:
        for cand in ["label","state","immobile","mobile","class","y","mobility","mobility status","mobility_status"]:
            if cand in df.columns:
                lkey = cand; break
        else:
            raise ValueError(f"Could not find a label-like column; got: {list(df.columns)}")

    s = df[lkey]

    # map to 0/1 (0=immobile, 1=mobile)
    if s.dtype == object:
        m = {
            "immobile":0, "im":0, "i":0, "0":0, "false":0, "f":0,
            "mobile":1, "m":1, "1":1, "true":1, "t":1
        }
        df["label"] = s.astype(str).str.strip().str.lower().map(m)
    else:
        v = pd.to_numeric(s, errors="coerce")
        if set(v.dropna().unique()) <= {0,1}:
            df["label"] = v.astype(int)
        elif set(v.dropna().unique()) <= {1,2}:
            # assume 1=immobile, 2=mobile; change here if your convention differs
            df["label"] = v.map({1:0, 2:1}).astype(int)
        else:
            df["label"] = (v > float(v.median())).astype(int)

    out = df[["t_start_sec","label"]].dropna().copy()
    out["t_start_sec"] = out["t_start_sec"].astype(float)
    out["label"] = out["label"].astype(int)
    out = out.sort_values("t_start_sec").reset_index(drop=True)
    out.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with {len(out)} rows.")
    
@app.command(name="from-persecond-batch")
def from_persecond_batch(
    glob_pattern: str = typer.Option("*.xlsx", help="Glob of Excel files, e.g., 'data/*.xlsx' or 'data/**/*.xlsx'"),
    sheet: str = typer.Option("Mobility_Status", help="Sheet name to read"),
    out_dir: str = typer.Option("labels", help="Output directory for labels CSVs"),
    time_col: str = typer.Option("Second", help="Time column name in the sheet"),
    label_col: str = typer.Option("Mobility Status", help="Label column name in the sheet"),
    recursive: bool = typer.Option(True, help="Enable ** recursive globs"),
):
    """
    Convert many Excel files at once (per-second sheet -> labels.csv).
    Skips Excel lock files that begin with '~$'.
    """
    import glob, os
    from rich import print
    os.makedirs(out_dir, exist_ok=True)

    files = glob.glob(glob_pattern, recursive=recursive)
    files = [f for f in files if not os.path.basename(f).startswith("~$")]
    if not files:
        raise typer.BadParameter(f"No files matched: {glob_pattern}")

    ok, fail = 0, 0
    for f in files:
        base = os.path.splitext(os.path.basename(f))[0]
        out_csv = os.path.join(out_dir, f"{base}.labels.csv")
        try:
            # Reuse the single-file converter with your column names
            from_persecond_sheet(
                xlsx_path=f, sheet=sheet, out_csv=out_csv,
                time_col=time_col, label_col=label_col
            )
            print(f"[green]OK[/green] {f} -> {out_csv}")
            ok += 1
        except Exception as e:
            print(f"[red]FAIL[/red] {f}: {e}")
            fail += 1

    print(f"[bold]Done.[/bold] Converted {ok}, failed {fail}. Output dir: {out_dir}")

if __name__ == "__main__":
    app()
