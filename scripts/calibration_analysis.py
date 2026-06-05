"""Calibration analysis for the for the unseen-ligand test subset.

or each model and each ligand (SMILES) with at least `min_interactions`
candidate GPCRs in the test set, we compute reliability diagrams, 
Expected Calibration Error (ECE), and Brier scores

Bins: 10 equal-width, [0.0, 0.1), [0.1, 0.2), ... [0.9, 1.0]. ECE is the
sample-weighted gap between mean predicted probability and observed
positive rate in each non-empty bin.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["mathtext.fontset"] = "dejavusans"
from sklearn.metrics import brier_score_loss


def _coerce_probs(series: pd.Series):
    """Return a float numpy array of probabilities.

    Some prediction files store Predictions_Proba as stringified lists
    (e.g. "[0.0727764]"). Match the project's historical parsing:
        float(str(x).strip("[]").split(",")[0])
    Falls back to a bare astype(float) when the column is already numeric.
    """
    try:
        return series.astype(float).to_numpy()
    except (TypeError, ValueError):
        pass
    try:
        return (series
                .apply(lambda x: float(str(x).strip("[]").split(",")[0]))
                .to_numpy())
    except (TypeError, ValueError):
        return None


# Supported filename patterns:
#   test_unseen_ligand_<model>_run<N>[_perm<MODE>_shuf<N>].csv
#   test_pred_unseen_lig_<model>_<N>.csv             (run number with no _run prefix)
#   val_pred_<model>_<N>.csv                         (etc.)
# Run number is optional; missing -> run=0.
PATTERN = re.compile(
    r"^(?P<dataset>"
    r"val_predictions|test_unseen_protein|test_unseen_ligand"
    r"|val_pred|test_pred_unseen_prot|test_pred_unseen_lig"
    r")_"
    r"(?P<model>.+?)"
    r"(?:_run(?P<run>\d+)|_(?P<run_short>\d+))?"
    r"(?:_perm(?P<perm>[A-Z]+)_shuf(?P<shuf>\d+))?\.csv$"
)

DATASET_ALIAS = {
    "val_pred": "val_predictions",
    "test_pred_unseen_prot": "test_unseen_protein",
    "test_pred_unseen_lig": "test_unseen_ligand",
}

# Map raw model directory/file names to the labels used in the manuscript.
MODEL_DISPLAY = {
    "cosine_ProtBert_MolFormer": "CosSim",
    "transformer_ProtBert_MolFormer": "Transformer",
    "xgb_ProtBert_MolFormer": "XGB",
    "fine_tune_none_ProtBert_MolFormer": "CA-Base",
    "fine_tune_only_mol_ProtBert_MolFormer": "CA-Lig",
    "fine_tune_only_prot_ProtBert_MolFormer": "CA-Prot",
    "fine_tune_both_ProtBert_MolFormer": "CA-Full",
}


# Manuscript-facing order for plot legends and summary tables.
LEGEND_ORDER = ["CosSim", "Transformer",
                "CA-Base", "CA-Lig", "CA-Prot", "CA-Full", "XGB"]


_PREFIX_DISPLAY = (
    ("fine_tune_none", "CA-Base"),
    ("fine_tune_only_mol", "CA-Lig"),
    ("fine_tune_only_prot", "CA-Prot"),
    ("fine_tune_both", "CA-Full"),
    ("transformer", "Transformer"),
    ("cosine", "CosSim"),
    ("xgboost", "XGB"),
    ("xgb", "XGB"),
)


def display_label(raw_model: str) -> str:
    """Map a raw model name to its manuscript display label.

    Looks up the exact name first, then falls back to a prefix match so
    embedding-suffixed variants (e.g. ``fine_tune_none_ProtBert_MolFormer``)
    still get the right label without needing every combination enumerated.
    """
    if raw_model in MODEL_DISPLAY:
        return MODEL_DISPLAY[raw_model]
    for prefix, label in _PREFIX_DISPLAY:
        if raw_model.startswith(prefix):
            return label
    return raw_model


def parse_name(name: str):
    m = PATTERN.match(name)
    if not m:
        return None
    ds = m.group("dataset")
    run = m.group("run") or m.group("run_short")
    return {
        "dataset": DATASET_ALIAS.get(ds, ds),
        "model": m.group("model"),
        "model_label": display_label(m.group("model")),
        "run": int(run) if run else 0,
        "permutation": m.group("perm") or "NONE",
        "shuffle_seed": int(m.group("shuf")) if m.group("shuf") else 0,
    }


def calibration_metrics(labels: np.ndarray, probs: np.ndarray, n_bins: int = 10):
    """Return ECE, Brier, and per-bin (mean_pred, pos_rate, count) arrays."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(probs, bin_edges[1:-1])      # 0..n_bins-1

    mean_pred = np.full(n_bins, np.nan)
    pos_rate = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)

    ece = 0.0
    n = len(labels)
    for b in range(n_bins):
        mask = bin_ids == b
        c = mask.sum()
        counts[b] = c
        if c == 0:
            continue
        mp = probs[mask].mean()
        pr = labels[mask].mean()
        mean_pred[b] = mp
        pos_rate[b] = pr
        ece += (c / n) * abs(mp - pr)

    brier = brier_score_loss(labels, probs)
    return float(ece), float(brier), mean_pred, pos_rate, counts, bin_edges


def reliability_plot(mean_pred, pos_rate, counts, bin_edges, title, out_path):
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(5.0, 5.2),
                                  gridspec_kw={"height_ratios": [3, 1]},
                                  sharex=True)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    ok = ~np.isnan(mean_pred)

    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="perfect calibration")
    ax.plot(mean_pred[ok], pos_rate[ok], "o-", color="tab:red",
            lw=1.5, ms=6, label="model")
    for x, y, c in zip(mean_pred[ok], pos_rate[ok], counts[ok]):
        ax.annotate(f"n={c}", (x, y), textcoords="offset points",
                    xytext=(5, -8), fontsize=7, color="gray")
    ax.set_ylabel("observed positive rate")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc="upper left", frameon=True, fontsize=8)
    ax.grid(alpha=0.3)

    ax2.bar(centers, counts, width=(bin_edges[1] - bin_edges[0]) * 0.9,
            color="tab:blue", alpha=0.7)
    ax2.set_xlabel("predicted probability")
    ax2.set_ylabel("# samples")
    ax2.set_xlim(0, 1)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def combined_reliability_plot(per_model_runs: dict, bin_edges, title, out_path,
                              show_band: bool = True):
    """Overlay models on a single reliability diagram, aggregating across runs.

    per_model_runs: dict[model_label] -> list of per-run dicts each containing
        keys mean_pred, pos_rate, counts, ece, brier, n, pos_rate_overall.
    For each model, bin statistics are averaged across runs (ignoring empty
    bins via nanmean); ECE / Brier in the legend are mean ± std across runs.
    """
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1,
            label="Perfect calibration")

    cmap = plt.get_cmap("tab10")
    order_idx = {name: i for i, name in enumerate(LEGEND_ORDER)}
    models_sorted = sorted(per_model_runs,
                           key=lambda m: (order_idx.get(m, len(LEGEND_ORDER)), m))
    for i, model in enumerate(models_sorted):
        runs = per_model_runs[model]
        mp_stack = np.vstack([r["mean_pred"] for r in runs])
        pr_stack = np.vstack([r["pos_rate"] for r in runs])
        with np.errstate(invalid="ignore"):
            mp_mean = np.nanmean(mp_stack, axis=0)
            pr_mean = np.nanmean(pr_stack, axis=0)
            pr_std = np.nanstd(pr_stack, axis=0)
        ok = ~np.isnan(mp_mean) & ~np.isnan(pr_mean)

        eces = np.array([r["ece"] for r in runs])
        briers = np.array([r["brier"] for r in runs])
        n_runs = len(runs)
        # Re-apply display_label defensively so anything that slipped past
        # parse_name (e.g. legacy bucket keys) still gets the manuscript name.
        label = f"{display_label(model)} (ECE={eces.mean():.2f})"

        color = cmap(i % 10)
        yerr = pr_std[ok] if (show_band and n_runs > 1) else None
        ax.errorbar(mp_mean[ok], pr_mean[ok], yerr=yerr,
                    fmt="o-", color=color, ecolor=color,
                    lw=1.6, ms=5, capsize=3, elinewidth=1,
                    label=label)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed positive rate")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
              fontsize=8, frameon=False, borderaxespad=0.0)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main(args):
    rdir = Path(args.results_dir)
    files = sorted(rdir.glob(args.glob))
    if not files:
        sys.exit(f"No files matched {args.glob!r} in {rdir}")

    rows = []
    skipped = []
    out_subdir = rdir / "calibration"
    out_subdir.mkdir(parents=True, exist_ok=True)

    combined = {}        # (dataset, perm) -> {model_label: per-bin dict}
    shared_edges = None

    for f in files:
        info = parse_name(f.name)
        if info is None:
            skipped.append(f.name); continue

        if info["permutation"] != "NONE":
            continue
        df = pd.read_csv(f)
        if "Label" not in df or "Predictions_Proba" not in df:
            skipped.append(f.name + " (missing columns)"); continue

        n_before = len(df)

        if args.min_ligand_interactions > 1:
            if "SMILES" not in df.columns:
                skipped.append(f.name + " (no SMILES column for selectivity filter)")
                continue
            counts_per_smiles = df["SMILES"].value_counts()
            keep_smiles = counts_per_smiles[
                counts_per_smiles >= args.min_ligand_interactions
            ].index
            df = df[df["SMILES"].isin(keep_smiles)].reset_index(drop=True)
            if len(df) == 0:
                skipped.append(f.name + " (no ligands met the >=N-interactions filter)")
                continue

        labels = df["Label"].astype(int).to_numpy()
        probs = _coerce_probs(df["Predictions_Proba"])
        if probs is None:
            skipped.append(f.name + " (could not parse Predictions_Proba)")
            continue
        if len(np.unique(labels)) < 2:
            skipped.append(f.name + " (one-class labels)"); continue

        ece, brier, mp, pr, counts, edges = calibration_metrics(
            labels, probs, n_bins=args.n_bins)

        subset_tag = (f"selectivity subset (n={len(df)}/{n_before}, "
                      f">={args.min_ligand_interactions} interactions/ligand)"
                      if args.min_ligand_interactions > 1
                      else f"full (n={len(df)})")
        title = (f"{info['model']} · {info['dataset']} · run{info['run']} · "
                 f"{info['permutation']}\n{subset_tag}\n"
                 f"ECE={ece:.2f}  Brier={brier:.2f}  "
                 f"pos_rate={labels.mean():.2f}")
        if not args.no_per_file_plots:
            out_png = out_subdir / f"reliability_{f.stem}.png"
            reliability_plot(mp, pr, counts, edges, title, out_png)

        rows.append({**info, "n": len(df),
                     "pos_rate": float(labels.mean()),
                     "ece": ece, "brier": brier,
                     "file": f.name})

        key = (info["dataset"], info["permutation"])
        model_buckets = combined.setdefault(key, {})
        model_buckets.setdefault(info["model_label"], []).append({
            "mean_pred": mp, "pos_rate": pr, "counts": counts,
            "ece": ece, "brier": brier,
            "n": len(df), "pos_rate_overall": float(labels.mean()),
            "run": info["run"],
        })
        shared_edges = edges

    if not rows:
        sys.exit("No valid prediction files were parsed.")

    raw = pd.DataFrame(rows)
    raw.to_csv(rdir / "calibration_per_run.csv", index=False)

    summary = (raw
               .groupby(["model_label", "dataset", "permutation"])
               [["ece", "brier", "pos_rate"]]
               .agg(["mean", "std", "count"])
               .round(2))
    summary.to_csv(rdir / "calibration_summary_raw.csv")

    order_idx = {n: i for i, n in enumerate(LEGEND_ORDER)}
    pretty_rows = []
    for (model, dataset, perm), block in raw.groupby(
            ["model_label", "dataset", "permutation"]):
        row = {"Model": model, "Dataset": dataset, "Permutation": perm}
        for metric, label in [("ece", "ECE"),
                              ("brier", "Brier"),
                              ("pos_rate", "Positive rate")]:
            m = float(block[metric].mean())
            s = float(block[metric].std())
            if metric == "pos_rate":
                row[label] = f"{m:.2f}"
            else:
                row[label] = f"{m:.2f} ± {s:.2f}"
        row["n_seeds"] = int(len(block))
        row["__ord__"] = order_idx.get(model, len(LEGEND_ORDER))
        pretty_rows.append(row)

    summary_pretty = (pd.DataFrame(pretty_rows)
                      .sort_values(["Dataset", "Permutation", "__ord__"])
                      .drop(columns="__ord__"))
    summary_pretty.to_csv(rdir / "calibration_summary.csv", index=False)

    # Combined reliability diagrams — one per (dataset, permutation) group.
    for (dataset, perm), models in combined.items():
        if not models:
            continue
        subset_tag = (f"selectivity subset (>= {args.min_ligand_interactions} "
                      f"interactions/ligand)"
                      if args.min_ligand_interactions > 1
                      else "full subset")
        title = (f"Reliability — {dataset} · perm={perm}\n{subset_tag}")
        out_png = out_subdir / f"reliability_combined_{dataset}_{perm}.png"
        combined_reliability_plot(models, shared_edges, title, out_png)

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    print(f"\nLigand-selectivity filter: >= {args.min_ligand_interactions} "
          f"interactions per ligand "
          f"({'disabled' if args.min_ligand_interactions <= 1 else 'enabled'})")
    print(f"Per-file metrics  -> {rdir / 'calibration_per_run.csv'}")
    print(f"Grouped summary   -> {rdir / 'calibration_summary.csv'}")
    print(f"Reliability plots -> {out_subdir}/")

    print("\n" + "=" * 70)
    print("CALIBRATION SUMMARY (ECE, Brier) by (model, dataset, permutation)")
    print("=" * 70)
    print(summary)

    print("\nInterpretation guide:")
    print("  ECE   < 0.05  ~  well calibrated")
    print("  ECE   0.05-0.15  mildly mis-calibrated")
    print("  ECE   > 0.15  poorly calibrated (typically over-confident)")
    print("  Brier  lower is better; compare against the no-skill baseline")
    print("         (= pos_rate * (1 - pos_rate)) which appears above.")

    if skipped:
        print(f"\nSkipped {len(skipped)} file(s):")
        for s in skipped[:20]:
            print(f"  - {s}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Calibration analysis (reliability diagrams + ECE + Brier)")
    p.add_argument("--results_dir", type=str, default="results")
    p.add_argument("--glob", type=str,
                   default="test_*unseen_li*_*.csv",
                   help="Glob pattern for prediction CSVs. Default matches "
                        "both 'test_unseen_ligand_*.csv' and "
                        "'test_pred_unseen_lig_*.csv' (the legacy naming). "
                        "Override to widen scope or restrict.")
    p.add_argument("--n_bins", type=int, default=10,
                   help="Number of equal-width probability bins (default 10)")
    p.add_argument("--min_ligand_interactions", type=int, default=5,
                   help="Restrict to ligands appearing in at least this many "
                        "rows of each file (the manuscript's ligand-selectivity "
                        "subset uses 5). Set to 1 to disable filtering.")
    p.add_argument("--no_per_file_plots", action="store_true",
                   help="Skip the per-file reliability PNGs and only write the "
                        "combined overlay figure(s).")
    args = p.parse_args()
    main(args)
