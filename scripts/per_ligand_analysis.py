"""Per-ligand selectivity analyses for the unseen-ligand test subset.

For each model and each ligand (SMILES) with at least `min_interactions`
candidate GPCRs in the test set, we compute three quantities:

  1. Per-ligand AUROC and AUPRC over its candidate receptors.
     Ligands with single-class labels are skipped (no AUROC defined).

  2. Top-k recovery of active receptors:
       Recall@k = (# true actives in top-k by predicted prob) / (# true actives)
     plus the mean reciprocal rank (MRR) of the first true active.

  3. Probability gap:
       delta = mean P(bind | label=1) - mean P(bind | label=0)
     per ligand, summarised across ligands.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score

from calibration_analysis import (
    parse_name, display_label, _coerce_probs,
)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["mathtext.fontset"] = "dejavusans"


LEGEND_ORDER = ["CosSim", "Transformer",
                "CA-Base", "CA-Lig", "CA-Prot", "CA-Full", "XGB"]


def per_ligand_metrics(df: pd.DataFrame, ks=(1, 3, 5)):
    """Return one row per ligand with AUROC, AUPRC, recall@k, MRR, delta.

    df must have columns: SMILES, Label (0/1), prob (float).
    """
    rows = []
    for smi, g in df.groupby("SMILES", sort=False):
        y = g["Label"].to_numpy().astype(int)
        p = g["prob"].to_numpy().astype(float)
        n = len(y)
        n_pos = int(y.sum())
        n_neg = int(n - n_pos)

        row = {"SMILES": smi, "n_receptors": n,
               "n_pos": n_pos, "n_neg": n_neg}

        if n_pos > 0 and n_neg > 0:
            row["auroc"] = roc_auc_score(y, p)
            row["auprc"] = average_precision_score(y, p)
            row["delta_prob"] = p[y == 1].mean() - p[y == 0].mean()
        else:
            row["auroc"] = np.nan
            row["auprc"] = np.nan
            row["delta_prob"] = np.nan

        order = np.argsort(-p, kind="mergesort")  # stable desc
        y_sorted = y[order]
        for k in ks:
            if n_pos > 0 and n >= k:
                row[f"recall_at_{k}"] = y_sorted[:k].sum() / n_pos
            else:
                row[f"recall_at_{k}"] = np.nan

        if n_pos > 0:
            first = np.argmax(y_sorted == 1)  # rank index of first hit
            row["mrr"] = 1.0 / (first + 1)
        else:
            row["mrr"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def summarise(per_lig: pd.DataFrame, metric_cols):
    """Across-ligand median, IQR, mean for one (model, run) frame."""
    out = {}
    for col in metric_cols:
        s = per_lig[col].dropna()
        if len(s) == 0:
            out[f"{col}_median"] = np.nan
            out[f"{col}_q1"] = np.nan
            out[f"{col}_q3"] = np.nan
            out[f"{col}_mean"] = np.nan
            out[f"{col}_n"] = 0
        else:
            out[f"{col}_median"] = float(s.median())
            out[f"{col}_q1"] = float(s.quantile(0.25))
            out[f"{col}_q3"] = float(s.quantile(0.75))
            out[f"{col}_mean"] = float(s.mean())
            out[f"{col}_n"] = int(len(s))
    return out


def cdf_plot(per_lig_by_model: dict, metric: str, xlabel: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    order_idx = {name: i for i, name in enumerate(LEGEND_ORDER)}
    cmap = plt.get_cmap("tab10")
    keys = sorted(per_lig_by_model,
                  key=lambda m: (order_idx.get(m, len(LEGEND_ORDER)), m))
    for i, model in enumerate(keys):
        vals = np.sort(per_lig_by_model[model][metric].dropna().to_numpy())
        if len(vals) == 0:
            continue
        y = np.arange(1, len(vals) + 1) / len(vals)
        med = np.median(vals)
        ax.plot(vals, y, lw=1.8, color=cmap(i % 10),
                label=f"{model} (median={med:.2f})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Cumulative fraction of ligands")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
              fontsize=8, frameon=False, borderaxespad=0.0)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main(args):
    rdir = Path(args.results_dir)
    files = sorted(rdir.glob(args.glob))
    files = [f for f in files if "unseen_lig" in f.name or "unseen_ligand" in f.name]
    if not files:
        sys.exit(f"No matching prediction files in {rdir}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_run_rows = []          # one row per (model, run): across-ligand summary
    per_lig_long_rows = []     # long-form: one row per (model, run, ligand)
    per_lig_by_model = {}      # model_label -> concatenated per-ligand frame (all runs)
    skipped = []

    metric_cols = ["auroc", "auprc", "delta_prob", "mrr",
                   "recall_at_1", "recall_at_3", "recall_at_5"]

    for f in files:
        info = parse_name(f.name)
        if info is None:
            skipped.append(f.name + " (unparseable name)"); continue

        df = pd.read_csv(f)
        if not {"SMILES", "Label", "Predictions_Proba"}.issubset(df.columns):
            skipped.append(f.name + " (missing required columns)"); continue

        n_before = len(df)
        if args.min_ligand_interactions > 1:
            counts = df["SMILES"].value_counts()
            keep = counts[counts >= args.min_ligand_interactions].index
            df = df[df["SMILES"].isin(keep)].reset_index(drop=True)
            if len(df) == 0:
                skipped.append(f.name + " (no ligands met the >=N filter)")
                continue

        probs = _coerce_probs(df["Predictions_Proba"])
        if probs is None:
            skipped.append(f.name + " (could not parse Predictions_Proba)")
            continue
        df = df.assign(prob=probs, Label=df["Label"].astype(int))

        per_lig = per_ligand_metrics(df)
        per_lig["model"] = info["model"]
        per_lig["model_label"] = display_label(info["model"])
        per_lig["run"] = info["run"]
        per_lig["n_rows_used"] = len(df)
        per_lig["n_rows_orig"] = n_before
        per_lig_long_rows.append(per_lig)

        summary = summarise(per_lig, metric_cols)
        summary.update({
            "model": info["model"],
            "model_label": display_label(info["model"]),
            "run": info["run"],
            "n_ligands": int(per_lig["SMILES"].nunique()),
            "n_rows_used": len(df),
            "n_rows_orig": n_before,
        })
        per_run_rows.append(summary)

        per_lig_by_model.setdefault(display_label(info["model"]),
                                    []).append(per_lig)

    if not per_run_rows:
        sys.exit("No usable prediction files after filtering. "
                 f"Skipped: {skipped}")

    per_run_df = pd.DataFrame(per_run_rows)
    per_run_df.to_csv(out_dir / "per_ligand_per_run.csv", index=False)

    long_df = pd.concat(per_lig_long_rows, ignore_index=True)
    long_df.to_csv(out_dir / "per_ligand_long.csv", index=False)

    # Across-seed summary: mean +/- std of the across-ligand statistics.
    agg_cols = [c for c in per_run_df.columns
                if c.endswith(("_median", "_mean", "_q1", "_q3"))]
    summary_df = (per_run_df
                  .groupby("model_label")[agg_cols]
                  .agg(["mean", "std"])
                  .round(2))
    summary_df.to_csv(out_dir / "per_ligand_summary.csv")

    # CDF plots (pooled across runs, one curve per model).
    pooled = {m: pd.concat(parts, ignore_index=True)
              for m, parts in per_lig_by_model.items()}
    cdf_plot(pooled, "auroc", "Per-ligand AUROC",
             out_dir / "cdf_per_ligand_auroc.png")
    cdf_plot(pooled, "auprc", "Per-ligand AUPRC",
             out_dir / "cdf_per_ligand_auprc.png")
    cdf_plot(pooled, "delta_prob",
             "Per-ligand probability gap  (mean P|active − mean P|inactive)",
             out_dir / "cdf_per_ligand_delta.png")

    headline_cols = ["auroc_mean", "auprc_mean", "delta_prob_mean",
                     "mrr_mean", "recall_at_1_mean",
                     "recall_at_3_mean", "recall_at_5_mean"]
    pretty_names = {
        "auroc_mean":       "AUROC",
        "auprc_mean":       "AUPRC",
        "delta_prob_mean":  "Delta (P_active - P_inactive)",
        "mrr_mean":         "MRR",
        "recall_at_1_mean": "Recall@1",
        "recall_at_3_mean": "Recall@3",
        "recall_at_5_mean": "Recall@5",
    }
    stats = (per_run_df
             .groupby("model_label")[headline_cols]
             .agg(["mean", "std"])
             .round(2))
    headline = pd.DataFrame(index=stats.index)
    for col in headline_cols:
        m = stats[(col, "mean")]
        s = stats[(col, "std")]
        headline[pretty_names[col]] = [f"{mm:.2f} ± {ss:.2f}"
                                       for mm, ss in zip(m, s)]
    order_idx = {n: i for i, n in enumerate(LEGEND_ORDER)}
    headline["__ord__"] = [order_idx.get(m, len(LEGEND_ORDER))
                           for m in headline.index]
    headline = headline.sort_values("__ord__").drop(columns="__ord__")
    headline.index.name = "Model"
    headline.to_csv(out_dir / "per_ligand_headline.csv")

    contrib = (per_run_df
               .groupby("model_label")[["auroc_n", "mrr_n", "n_ligands"]]
               .first())
    print("\nPer-ligand selectivity summary "
          f"(>= {args.min_ligand_interactions} interactions/ligand)\n"
          + "=" * 72)
    print(f"Ligands in subset:                 {int(contrib['n_ligands'].iloc[0])}")
    print(f"Contributing to AUROC/AUPRC/Delta: {int(contrib['auroc_n'].iloc[0])} "
          "(both classes present)")
    print(f"Contributing to Recall@k / MRR:    {int(contrib['mrr_n'].iloc[0])} "
          "(>= 1 active receptor)\n")
    print(headline.to_string())
    print()
    if skipped:
        print("Skipped files:")
        for s in skipped:
            print(" ", s)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results",
                    help="Directory containing test_pred_unseen_lig_*.csv")
    ap.add_argument("--out_dir", default="per_ligand_analysis",
                    help="Output directory for CSVs and CDF plots.")
    ap.add_argument("--glob", default="test_pred_unseen_lig_*.csv",
                    help="Glob for prediction files inside --results_dir.")
    ap.add_argument("--min_ligand_interactions", type=int, default=5,
                    help="Same ligand-selectivity filter as the "
                         "calibration analysis (default 5).")
    main(ap.parse_args())
