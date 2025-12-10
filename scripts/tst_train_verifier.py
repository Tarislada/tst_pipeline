# scripts/tst_train_verifier.py
import os, glob, json
from typing import Optional, List
import pandas as pd
import numpy as np
import typer
from rich import print
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib

app = typer.Typer(help="Train the verifier (Stage-B)")

CONTROL_COLS = {
    "t_start_sec", "label", "rule_pred", "is_uncertain",
    # engineerd signals you may want to keep as features; remove from here if desired:
    # "area_mean","area_std","sim_mean","sim_std","pwr_lo_0_2hz","pwr_lo_0_5hz",
}

def _base_no_suffix(path: str) -> str:
    b = os.path.basename(path)
    for suf in (".features.csv", ".merged.csv", ".labels.csv"):
        if b.endswith(suf):
            return b[: -len(suf)]
    return os.path.splitext(b)[0]

def _numeric_feature_cols(df: pd.DataFrame) -> List[str]:
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    # keep numeric, drop control columns
    return [c for c in num if c not in CONTROL_COLS]

def _load_join_one(features_path: str, labels_map: dict, require_uncertain: bool) -> pd.DataFrame:
    base = _base_no_suffix(features_path)
    labels_path = labels_map.get(base)
    if labels_path is None:
        print(f"[yellow]SKIP[/yellow] no labels for base '{base}'")
        return pd.DataFrame()

    f = pd.read_csv(features_path)
    l = pd.read_csv(labels_path)
    # inner join → only overlapping seconds
    m = f.merge(l, on="t_start_sec", how="inner", suffixes=("", "_lab"))
    if "label" not in m.columns:
        raise RuntimeError(f"{labels_path} missing 'label'")

    if require_uncertain:
        if "is_uncertain" not in m.columns:
            raise RuntimeError(f"{features_path} missing 'is_uncertain' column. "
                               "Re-run feature extraction with threshold/band or call with --no-only-uncertain.")
        m = m[m["is_uncertain"] == 1]

    m["__base__"] = base
    return m

@app.command("train-batch")
def train_batch(
    features_glob: str = typer.Option(..., help="e.g., 'data/merged/*.merged.csv' or 'data/features/*.features.csv'"),
    labels_glob: str = typer.Option(..., help="e.g., 'labels/*.labels.csv'"),
    model_out: str = typer.Option("verifier_rf.joblib"),
    only_uncertain: bool = typer.Option(True, help="Train only on rows flagged is_uncertain=1"),
    test_size: float = typer.Option(0.2),
    random_state: int = typer.Option(42),
    n_estimators: int = typer.Option(400),
    max_depth: Optional[int] = typer.Option(None),
    n_jobs: int = typer.Option(-1),
):
    """Train a single verifier on (optionally) all uncertain windows pooled across sessions."""
    f_paths = sorted(glob.glob(features_glob))
    l_paths = sorted(glob.glob(labels_glob))
    if not f_paths:
        raise typer.BadParameter(f"No features matched: {features_glob}")
    if not l_paths:
        raise typer.BadParameter(f"No labels matched: {labels_glob}")

    labels_map = {_base_no_suffix(p): p for p in l_paths}

    # build dataset
    parts = []
    kept, skipped = 0, 0
    for fp in f_paths:
        df = _load_join_one(fp, labels_map, require_uncertain=only_uncertain)
        if df.empty:
            skipped += 1
            continue
        parts.append(df)
        kept += 1
    if not parts:
        raise RuntimeError("No data rows found after joining. Check naming or alignment.")
    data = pd.concat(parts, axis=0, ignore_index=True)

    # X / y
    feat_cols = _numeric_feature_cols(data)
    if not feat_cols:
        raise RuntimeError("No numeric feature columns found.")
    X = data[feat_cols].values
    y = data["label"].astype(int).values

    # split by rows (simple). If you prefer group-wise split by session, we can switch to GroupKFold on '__base__'.
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if len(np.unique(y))>1 else None
    )

    clf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, n_jobs=n_jobs,
        class_weight="balanced_subsample", random_state=random_state
    )
    clf.fit(X_tr, y_tr)

    y_pr = clf.predict(X_te)
    try:
        y_ps = clf.predict_proba(X_te)[:,1]
        auc = roc_auc_score(y_te, y_ps)
    except Exception:
        auc = float("nan")

    print("[bold]Report (holdout):[/bold]")
    print(classification_report(y_te, y_pr, digits=3))
    print(f"AUC: {auc:.4f}")

    # save model + metadata
    os.makedirs(os.path.dirname(model_out) or ".", exist_ok=True)
    joblib.dump(clf, model_out)
    meta = {
        "features_glob": features_glob,
        "labels_glob": labels_glob,
        "only_uncertain": only_uncertain,
        "test_size": test_size,
        "random_state": random_state,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "feature_columns": feat_cols,
        "rows_total": int(len(data)),
        "sessions_used": int(data["__base__"].nunique()),
        "sessions_kept": kept,
        "sessions_skipped": skipped,
    }
    with open(model_out + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[green]Saved[/green] {model_out}  (+ meta JSON)")

if __name__ == "__main__":
    app()
