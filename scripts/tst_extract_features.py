import typer
from rich import print
from tst.config import PipelineConfig
from tst.pipeline import TSTPipeline

app = typer.Typer()

@app.command()
def extract(
    input_npy: str = typer.Option("output_data.npy", help="Path to npy with area or [area,sim]."),
    out_csv: str = typer.Option("tst_features_persec.csv", help="Output features CSV."),
    fps: int = typer.Option(30),
    win_s: float = typer.Option(1.0),
    hop_s: float = typer.Option(None),
    beta: float = typer.Option(0.31),
    w_min: float = typer.Option(0.75),
    threshold: float = typer.Option(0.02),
    band: float = typer.Option(0.30),
    norm_within_video: bool = typer.Option(True),
    add_lowfreq_power: bool = typer.Option(True),
):
    cfg = PipelineConfig()
    cfg.paths.input_npy = input_npy
    cfg.paths.out_features_csv = out_csv
    cfg.window.fps = fps
    cfg.window.win_s = win_s
    cfg.window.hop_s = hop_s
    cfg.rule.beta = beta
    cfg.rule.w_min = w_min
    cfg.rule.threshold = threshold
    cfg.rule.band = band
    cfg.norm_within_video = norm_within_video
    cfg.add_lowfreq_power = add_lowfreq_power

    pipe = TSTPipeline(cfg)
    df = pipe.extract_features()
    print(f"[green]Saved features:[/green] {out_csv}, rows={len(df)}")
    
@app.command(name="extract-batch")
def extract_batch(
    glob_pattern: str = typer.Option("npy/*.npy", help="Glob of .npy files"),
    out_dir: str = typer.Option("features", help="Output directory for per-video features"),
    fps: int = typer.Option(30),
    win_s: float = typer.Option(1.0),
    hop_s: float = typer.Option(None),
    beta: float = typer.Option(0.31),
    w_min: float = typer.Option(0.75),
    threshold: float = typer.Option(0.02),
    band: float = typer.Option(0.30),
    norm_within_video: bool = typer.Option(True),
    add_lowfreq_power: bool = typer.Option(True),
):
    """
    Run feature extraction for many .npy files at once.
    Each .npy should be shape (T,) for area or (T,2) for [area, similarity].
    """
    import glob, os
    from rich import print
    os.makedirs(out_dir, exist_ok=True)
    files = [p for p in glob.glob(glob_pattern) if p.lower().endswith(".npy")]
    if not files:
        raise typer.BadParameter(f"No files matched: {glob_pattern}")

    from tst.config import PipelineConfig
    from tst.pipeline import TSTPipeline

    ok, fail = 0, 0
    for f in files:
        base = os.path.splitext(os.path.basename(f))[0]
        out_csv = os.path.join(out_dir, f"{base}.features.csv")
        try:
            cfg = PipelineConfig()
            cfg.paths.input_npy = f
            cfg.paths.out_features_csv = out_csv
            cfg.window.fps = fps
            cfg.window.win_s = win_s
            cfg.window.hop_s = hop_s
            cfg.rule.beta = beta
            cfg.rule.w_min = w_min
            cfg.rule.threshold = threshold
            cfg.rule.band = band
            cfg.norm_within_video = norm_within_video
            cfg.add_lowfreq_power = add_lowfreq_power

            pipe = TSTPipeline(cfg)
            df = pipe.extract_features()
            print(f"[green]OK[/green] {f} -> {out_csv} (rows={len(df)})")
            ok += 1
        except Exception as e:
            print(f"[red]FAIL[/red] {f}: {e}")
            fail += 1
    print(f"[bold]Done.[/bold] Converted {ok}, failed {fail}. Output dir: {out_dir}")

if __name__ == "__main__":
    app()
