"""Representation-level analyses for the cross-attention DTI checkpoints
(protein side only).

Compares each fine-tuned ProtBert encoder against the pretrained
`Rostlab/prot_bert` baseline:

  1. Per-receptor embedding drift (cosine + L2 distance)
  2. Joint UMAP / t-SNE visualization
  3. Variance / representation-collapse diagnostics
  4. Layer-wise relative Frobenius weight drift inside ProtBert
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import gc
import json
import warnings
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["mathtext.fontset"] = "dejavusans"
from pathlib import Path
from typing import Optional, Tuple
from transformers import AutoModel, AutoTokenizer

from src.utils.data_loader import load_datasets

SUBFAMILY_MAPPING_CSV = Path(__file__).resolve().parent.parent / \
    "data" / "uniprot_subfamily.csv"

# UMAP forces n_jobs=1 when random_state is set (we want reproducibility);
# the resulting UserWarning is informational and clutters the analysis log.
warnings.filterwarnings(
    "ignore",
    message=r"n_jobs value .* overridden to 1 by setting random_state.*",
    category=UserWarning,
)


# ---------------------------------------------------------------------------
# Encoder loading
# ---------------------------------------------------------------------------

def load_baseline_encoder(name: str, device):
    return AutoModel.from_pretrained(name).to(device).eval()


def load_finetuned_encoder(ckpt_path: Path, device, protein_model: str):
    """Load the ProtBert encoder out of a CA checkpoint.

    Instantiates a bare HuggingFace AutoModel, filters the checkpoint
    state_dict down to the ``protein_encoder.*`` keys, and loads them with
    strict=False so HF-version-dependent buffers (position_ids etc.) are
    tolerated.
    """
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    target = AutoModel.from_pretrained(protein_model).to(device).eval()
    prefix = "protein_encoder."
    filtered = {k[len(prefix):]: v for k, v in state.items()
                if k.startswith(prefix)}
    missing, _ = target.load_state_dict(filtered, strict=False)
    weight_missing = [k for k in missing
                      if not k.endswith(("position_ids", "token_type_ids"))]
    if weight_missing:
        raise RuntimeError(
            f"Missing protein-encoder weights in {ckpt_path.name}: "
            f"{weight_missing[:5]}{'...' if len(weight_missing) > 5 else ''}")
    return target


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def embed_sequences(encoder, tokenizer, sequences, max_len, device, batch_size=4):
    """Return (mean_pooled, cls_pooled) of shape (n_seqs, hidden_size).

    Uses the ProtBert convention of whitespace-separated amino acids and
    fixed max_length padding.
    """
    mean_pool, cls_pool = [], []
    for i in range(0, len(sequences), batch_size):
        batch = [" ".join(list(s)) for s in sequences[i:i + batch_size]]
        tok = tokenizer(batch, max_length=max_len, padding="max_length",
                        truncation=True, return_tensors="pt").to(device)
        out = encoder(input_ids=tok["input_ids"],
                      attention_mask=tok["attention_mask"])
        hidden = out.last_hidden_state
        mask = tok["attention_mask"].unsqueeze(-1).float()
        mean_vec = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        cls_vec = hidden[:, 0, :]
        mean_pool.append(mean_vec.cpu().numpy())
        cls_pool.append(cls_vec.cpu().numpy())
    return np.vstack(mean_pool), np.vstack(cls_pool)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def cosine_distance(a, b):
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return 1.0 - (an * bn).sum(1)


def collapse_metrics(emb: np.ndarray) -> dict:
    n = emb.shape[0]
    e = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    sim = e @ e.T
    iu = np.triu_indices(n, k=1)
    mean_pairwise_cos_dist = float(1.0 - sim[iu].mean())
    centered = emb - emb.mean(0, keepdims=True)
    svals = np.linalg.svd(centered, compute_uv=False)
    total_var = float((svals ** 2).sum() / max(n - 1, 1))
    norm = svals / (svals.sum() + 1e-12)
    effective_rank = float(np.exp(-(norm * np.log(norm + 1e-12)).sum()))
    return {"mean_pairwise_cos_dist": mean_pairwise_cos_dist,
            "total_variance": total_var,
            "effective_rank": effective_rank,
            "n": n}


def layerwise_drift(baseline_encoder, finetuned_encoder) -> pd.DataFrame:
    base = dict(baseline_encoder.state_dict())
    rows = []
    for name, ft_w in finetuned_encoder.state_dict().items():
        if name not in base:
            continue
        if ft_w.dtype not in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
            continue
        b = base[name].float()
        f = ft_w.float()
        denom = b.norm().item()
        if denom == 0:
            continue
        rows.append({"param": name,
                     "rel_frob": (f - b).norm().item() / denom,
                     "abs_frob": (f - b).norm().item(),
                     "base_frob": denom})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# UMAP / t-SNE plot
# ---------------------------------------------------------------------------

def fit_2d(baseline_emb, finetuned_emb):
    """Fit one UMAP (or t-SNE fallback) on the stacked embeddings and return
    the 2D coords for the pretrained and fine-tuned halves separately. Doing
    a single fit keeps the two layouts directly comparable."""
    try:
        import umap
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    except ImportError:
        from sklearn.manifold import TSNE
        print("  (umap-learn not installed, falling back to t-SNE)")
        reducer = TSNE(n_components=2, perplexity=min(30, len(baseline_emb) - 1),
                       random_state=42, init="pca")
    stacked = np.vstack([baseline_emb, finetuned_emb])
    proj = reducer.fit_transform(stacked)
    n = len(baseline_emb)
    return proj[:n], proj[n:]


def umap_plot(b_xy, f_xy, out_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(b_xy[:, 0], b_xy[:, 1], s=18, alpha=0.7, label="Pretrained", c="tab:blue")
    ax.scatter(f_xy[:, 0], f_xy[:, 1], s=18, alpha=0.7, label="Fine-tuned", c="tab:red")
    for i in range(len(b_xy)):
        ax.plot([b_xy[i, 0], f_xy[i, 0]], [b_xy[i, 1], f_xy[i, 1]],
                "-", lw=0.4, alpha=0.35, color="gray")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _add_shape_handles(ax, existing_legend_kwargs):
    from matplotlib.lines import Line2D
    handles, lbls = ax.get_legend_handles_labels()
    kw = dict(existing_legend_kwargs)
    kw.pop("title", None)
    kw.pop("title_fontsize", None)

    # 1. Shape-key legend at upper-left, in a light-grey box.
    shape_handles = [
        Line2D([0], [0], marker="o", linestyle="none",
               markerfacecolor="lightgray", markeredgecolor="black",
               markersize=7, label="Pretrained"),
        Line2D([0], [0], marker="^", linestyle="none",
               markerfacecolor="lightgray", markeredgecolor="black",
               markersize=7, label="Fine-tuned"),
    ]
    shape_leg = ax.legend(
        handles=shape_handles, loc="upper left",
        fontsize=8, frameon=True, edgecolor="0.7", framealpha=0.95,
        handletextpad=0.5, borderpad=0.4,
    )
    try:
        shape_leg._legend_box.align = "left"
    except AttributeError:
        pass
    ax.add_artist(shape_leg) 

    kw_sub = dict(kw)
    kw_sub.setdefault("loc", "upper left")
    main_leg = ax.legend(handles, lbls, **kw_sub)
    try:
        main_leg._legend_box.align = "left"
    except AttributeError:
        pass
    return shape_leg, main_leg


def umap_subfamily_plot(b_xy, f_xy, labels, out_path):
    labels = list(labels)
    classes = sorted(set(labels))
    cmap = plt.get_cmap("tab20")
    color_for = {c: cmap(i % 20) for i, c in enumerate(classes)}

    fig, ax = plt.subplots(figsize=(7.5, 6))
    for i in range(len(b_xy)):
        ax.plot([b_xy[i, 0], f_xy[i, 0]], [b_xy[i, 1], f_xy[i, 1]],
                "-", lw=0.4, alpha=0.3, color="lightgray")
    for cls in classes:
        m = np.array([l == cls for l in labels])
        n = int(m.sum())
        ax.scatter(b_xy[m, 0], b_xy[m, 1], s=22, alpha=0.85,
                   marker="o", color=color_for[cls],
                   edgecolor="black", linewidth=0.3,
                   label=f"{cls} (n={n})")
        ax.scatter(f_xy[m, 0], f_xy[m, 1], s=22, alpha=0.85,
                   marker="^", color=color_for[cls],
                   edgecolor="black", linewidth=0.3)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    _add_shape_handles(ax, dict(loc="upper left", fontsize=8, frameon=False))
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def umap_centroid_plot(b_xy, f_xy, labels, out_path):
    """One arrow per subfamily, from its pretrained centroid to its
    fine-tuned centroid. Marker size scales with class count."""
    labels = list(labels)
    classes = sorted(set(labels))
    cmap = plt.get_cmap("tab20")

    fig, ax = plt.subplots(figsize=(7.5, 6))
    for i, cls in enumerate(classes):
        m = np.array([l == cls for l in labels])
        n = int(m.sum())
        if n < 1:
            continue
        b_c = b_xy[m].mean(0); f_c = f_xy[m].mean(0)
        color = cmap(i % 20)
        size = 60 + 18 * n
        ax.scatter(*b_c, s=size, marker="o", color=color,
                   edgecolor="black", linewidth=0.6, alpha=0.9,
                   label=f"{cls} (n={n})")
        ax.scatter(*f_c, s=size, marker="^", color=color,
                   edgecolor="black", linewidth=0.6, alpha=0.9)
        ax.annotate("", xy=f_c, xytext=b_c,
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.6))
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    _add_shape_handles(ax, dict(loc="upper left", fontsize=8, frameon=False))
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def subfamily_drift(baseline_emb, finetuned_emb, labels):
    """Per-class drift in the original embedding space.

    Returns a DataFrame with one row per subfamily and columns:
      n_seqs, centroid_cos_dist (between class-mean vectors),
      centroid_l2, mean_within_cos, mean_within_l2.
    """
    labels = np.asarray(labels)
    rows = []
    for cls in sorted(set(labels.tolist())):
        m = labels == cls
        n = int(m.sum())
        if n < 1:
            continue
        b = baseline_emb[m]
        f = finetuned_emb[m]
        b_c = b.mean(0); f_c = f.mean(0)
        denom_c = float(np.linalg.norm(b_c) * np.linalg.norm(f_c) + 1e-12)
        centroid_cos = 1.0 - float(b_c @ f_c) / denom_c
        centroid_l2 = float(np.linalg.norm(b_c - f_c))
        denom = np.linalg.norm(b, axis=1) * np.linalg.norm(f, axis=1) + 1e-12
        within_cos = 1.0 - (b * f).sum(1) / denom
        within_l2 = np.linalg.norm(b - f, axis=1)
        rows.append({"subfamily": cls,
                     "n_seqs": n,
                     "centroid_cos_dist": centroid_cos,
                     "centroid_l2": centroid_l2,
                     "mean_within_cos": float(within_cos.mean()),
                     "mean_within_l2": float(within_l2.mean())})
    return pd.DataFrame(rows)


def collect_seq_to_accession(config, sequence_column="Target Sequence",
                             accession_column="UniProt"):
    """sequence -> UniProt accession (first seen wins). Skips rows missing
    either field. Returns {} if the accession column is absent."""
    train_df, val_df, t1, t2 = load_datasets(
        data_dir=config["data"]["data_dir"],
        train_file=config["data"]["train_file"],
        val_file=config["data"]["val_file"],
        test_unseen_prot_file=config["data"]["test_unseen_protein"],
        test_unseen_lig_file=config["data"]["test_unseen_ligand"],
    )
    seq2acc = {}
    for d in (train_df, val_df, t1, t2):
        if accession_column not in d.columns or sequence_column not in d.columns:
            continue
        sub = d[[sequence_column, accession_column]].dropna()
        for seq, acc in zip(sub[sequence_column], sub[accession_column]):
            seq2acc.setdefault(seq, acc)
    return seq2acc


def load_subfamily_labels(sequences, seq2acc,
                          mapping_csv: Path = SUBFAMILY_MAPPING_CSV):
    """Return broad-subfamily labels aligned with `sequences`.
    """
    if not mapping_csv.exists():
        print(f"  Subfamily mapping CSV not found at {mapping_csv}; "
              "all sequences labelled Unclassified.")
        return ["Unclassified"] * len(sequences)
    sub_map = (pd.read_csv(mapping_csv)
                 .set_index("uniprot_id")["subfamily"].to_dict())
    labels = []
    missing_uniprot = []          # sequence had no accession at all
    missing_in_mapping = []       # UniProt present but not in mapping CSV
    for s in sequences:
        acc = seq2acc.get(s)
        if not acc:
            missing_uniprot.append(s[:15] + "...")
            labels.append("Unclassified")
            continue
        sub = sub_map.get(acc)
        if sub is None:
            missing_in_mapping.append(acc)
            labels.append("Unclassified")
            continue
        labels.append(sub)

    if missing_uniprot:
        print(f"  [Unclassified diag] {len(missing_uniprot)} sequence(s) had "
              "no UniProt in the train/val/test CSVs.")
        for s in missing_uniprot[:10]:
            print(f"    seq: {s}")
        if len(missing_uniprot) > 10:
            print(f"    ... and {len(missing_uniprot) - 10} more.")
    if missing_in_mapping:
        print(f"  [Unclassified diag] {len(missing_in_mapping)} UniProt "
              f"accession(s) NOT found in {mapping_csv.name} (add a row "
              "with the appropriate subfamily):")
        for a in missing_in_mapping[:20]:
            print(f"    {a}")
        if len(missing_in_mapping) > 20:
            print(f"    ... and {len(missing_in_mapping) - 20} more.")
    return labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def collect_sequences(config, column: str = "Target Sequence"):
    """Return (ordered_sequences, group_of_each_sequence) for the given column.
    """
    train_df, val_df, t1, t2 = load_datasets(
        data_dir=config["data"]["data_dir"],
        train_file=config["data"]["train_file"],
        val_file=config["data"]["val_file"],
        test_unseen_prot_file=config["data"]["test_unseen_protein"],
        test_unseen_lig_file=config["data"]["test_unseen_ligand"],
    )
    train_seqs = set(train_df[column].dropna().unique()) \
               | set(val_df[column].dropna().unique())
    unseen_prot_seqs = set(t1[column].dropna().unique()) - train_seqs
    unseen_lig_seqs = set(t2[column].dropna().unique()) - train_seqs - unseen_prot_seqs

    group = {}
    for s in train_seqs:        group[s] = "train"
    for s in unseen_prot_seqs:  group[s] = "unseen_protein"
    for s in unseen_lig_seqs:   group[s] = "unseen_ligand"

    sequences = sorted(group.keys())
    groups = [group[s] for s in sequences]
    return sequences, groups


def infer_variant(stem: str) -> str:
    """Map a checkpoint filename stem to a human-readable variant label."""
    if "prot_train_mol_frozen" in stem:
        return "CA-Prot"
    if "prot_train_mol_train" in stem:
        return "CA-Full"
    if "prot_frozen_mol_train" in stem:
        return "CA-Lig"
    if "prot_frozen_mol_frozen" in stem:
        return "CA-Base"
    return "unknown"

PROTEIN_CFG = {
    "column": "Target Sequence",
    "default_glob": "cross_attention_prot_train_*_run*.pth",
    "label": "ProtBert / GPCR",
    "max_len_default": 1024,
    "expected_variants": {"CA-Prot", "CA-Full"},
}


def analyze_side(args, config, device, out_dir: Path):
    """Run the full protein-side representation analysis."""
    side = "protein"
    cfg = PROTEIN_CFG
    print("\n" + "#" * 70)
    print(f"### ANALYSIS SIDE: PROTEIN ({cfg['label']})")
    print("#" * 70)

    sequences, groups = collect_sequences(config, column=cfg["column"])
    n_by_group = {g: groups.count(g) for g in set(groups)}
    print(f"\nAnalyzing {len(sequences)} unique GPCR entries:")
    for g, n in sorted(n_by_group.items()):
        print(f"  {g}: {n}")

    protein_model = args.protein_model
    baseline_name = protein_model
    max_len = args.max_len if args.max_len else cfg["max_len_default"]

    # ---- Find checkpoints ----
    results_dir = Path(args.results_dir or config["output"]["results_dir"])
    glob_pattern = args.glob if args.glob else cfg["default_glob"]
    ckpts = sorted(results_dir.glob(glob_pattern))
    if not ckpts:
        print(f"\nNo checkpoints matched {glob_pattern!r} under {results_dir}")
        return
    keep = [c for c in ckpts if infer_variant(c.stem) in cfg["expected_variants"]]
    if not keep:
        print(f"\nNo checkpoints with a trainable protein encoder matched "
              f"{glob_pattern!r}. Expected variants: {cfg['expected_variants']}")
        return
    ckpts = keep
    print(f"\nFound {len(ckpts)} checkpoint(s) with trainable protein encoder:")
    for c in ckpts:
        print(f"  - {c.name}  [{infer_variant(c.stem)}]")

    # ---- Baseline embeddings (computed once) ----
    print(f"\nLoading pretrained {baseline_name}...")
    tokenizer = AutoTokenizer.from_pretrained(baseline_name, do_lower_case=False)
    baseline = load_baseline_encoder(baseline_name, device)
    base_mean, base_cls = embed_sequences(
        baseline, tokenizer, sequences,
        max_len=max_len, device=device, batch_size=args.batch_size,
    )

    # Selected pooling strategies. Default is mean-only (stronger signal for
    # representation collapse, standard for protein representation analysis).
    pool_choices = {"mean": "mean", "cls": "cls", "both": "both"}[args.pooling]
    if pool_choices == "both":
        active_pools = (("mean", base_mean), ("cls", base_cls))
    elif pool_choices == "mean":
        active_pools = (("mean", base_mean),)
    else:
        active_pools = (("cls", base_cls),)
    print(f"Pooling strategy: {pool_choices}")

    drift_rows, collapse_rows, layer_rows = [], [], []
    variants_seen = set()

    # Broad GPCR subfamily labels — used to colour UMAPs and to compute
    # per-class centroid drift in the original embedding space.
    subfamily_drift_rows = []
    seq2acc = collect_seq_to_accession(config)
    subfamily_labels = load_subfamily_labels(sequences, seq2acc)
    n_unc = sum(1 for l in subfamily_labels if l == "Unclassified")
    from collections import Counter
    print(f"\nSubfamily labels assigned "
          f"({len(subfamily_labels) - n_unc}/{len(subfamily_labels)} resolved):")
    for cls, cnt in Counter(subfamily_labels).most_common():
        print(f"  {cls}: {cnt}")

    # Reference rows for collapse metrics — per pooling and per group, so
    # 'overall' / 'train' / 'unseen_protein' / 'unseen_ligand' can be compared.
    grp_arr = np.array(groups)
    def _per_group_collapse(checkpoint, variant_label, pool, emb):
        rows = [{"variant": variant_label, "checkpoint": checkpoint,
                 "pooling": pool, "group": "overall",
                 **collapse_metrics(emb)}]
        for g in ("train", "unseen_protein", "unseen_ligand"):
            m = grp_arr == g
            if m.sum() >= 2:                     # collapse metrics need >= 2 points
                rows.append({"variant": variant_label, "checkpoint": checkpoint,
                             "pooling": pool, "group": g,
                             **collapse_metrics(emb[m])})
        return rows

    for pool, emb in active_pools:
        collapse_rows += _per_group_collapse("pretrained", "pretrained", pool, emb)

    for ckpt in ckpts:
        # Per-checkpoint variant: explicit override wins; otherwise infer.
        variant = args.variant_name or infer_variant(ckpt.stem)
        variants_seen.add(variant)
        print(f"\n=== {ckpt.name}  [{variant}] ===")
        ft_encoder = load_finetuned_encoder(ckpt, device, protein_model)

        ft_mean, ft_cls = embed_sequences(
            ft_encoder, tokenizer, sequences,
            max_len=max_len, device=device, batch_size=args.batch_size,
        )
        ft_by_pool = {"mean": ft_mean, "cls": ft_cls}

        for pool, base_emb in active_pools:
            ft_emb = ft_by_pool[pool]
            cos = cosine_distance(base_emb, ft_emb)
            l2 = np.linalg.norm(base_emb - ft_emb, axis=1)
            for seq, grp, c, l in zip(sequences, groups, cos, l2):
                drift_rows.append({"variant": variant,
                                   "checkpoint": ckpt.stem,
                                   "pooling": pool,
                                   "group": grp,
                                   "sequence_hash": hash(seq),
                                   "cos_dist": float(c),
                                   "l2_dist": float(l)})
            # Per-group summary (helps separate train-set drift from
            # unseen-protein drift in the printout).
            grp_arr = np.array(groups)
            parts = []
            for g in ("train", "unseen_protein", "unseen_ligand"):
                m = grp_arr == g
                if m.any():
                    parts.append(f"{g}: cos={cos[m].mean():.4f}±{cos[m].std():.4f}")
            print(f"  drift [{pool}]  " + "  |  ".join(parts))

            collapse_rows += _per_group_collapse(ckpt.stem, variant, pool, ft_emb)

            fig, ax = plt.subplots(figsize=(5, 3.2))
            ax.hist(cos, bins=30, color="tab:red", alpha=0.85)
            ax.set_xlabel("Cosine distance (pretrained ↔ fine-tuned)")
            ax.set_ylabel("Number of GPCRs")
            fig.tight_layout()
            fig.savefig(out_dir / f"drift_hist_{variant}_{ckpt.stem}_{pool}.png", dpi=160)
            plt.close(fig)

            # UMAP is restricted to the training receptors only

            train_mask = np.array(groups) == "train"
            base_emb_tr = base_emb[train_mask]
            ft_emb_tr = ft_emb[train_mask]
            subfamily_labels_tr = [l for l, m in zip(subfamily_labels, train_mask) if m]

            b_xy, f_xy = fit_2d(base_emb_tr, ft_emb_tr)

            # Save UMAP coordinates + labels 
            np.savez(
                out_dir / f"umap_data_{variant}_{ckpt.stem}_{pool}.npz",
                b_xy=b_xy,
                f_xy=f_xy,
                labels=np.array(subfamily_labels_tr, dtype=object),
            )

            umap_plot(b_xy, f_xy,
                      out_path=out_dir / f"umap_{variant}_{ckpt.stem}_{pool}.png")
            umap_subfamily_plot(b_xy, f_xy, subfamily_labels_tr,
                out_path=out_dir / f"umap_subfamily_{variant}_{ckpt.stem}_{pool}.png")
            umap_centroid_plot(b_xy, f_xy, subfamily_labels_tr,
                out_path=out_dir / f"umap_subfamily_centroids_{variant}_{ckpt.stem}_{pool}.png")
            sf = subfamily_drift(base_emb, ft_emb, subfamily_labels)
            sf.insert(0, "pooling", pool)
            sf.insert(0, "checkpoint", ckpt.stem)
            sf.insert(0, "variant", variant)
            subfamily_drift_rows.append(sf)

        layer_df = layerwise_drift(baseline, ft_encoder)
        layer_df.insert(0, "checkpoint", ckpt.stem)
        layer_df.insert(0, "variant", variant)
        layer_rows.append(layer_df)

        layer_df_local = layer_df.copy()
        layer_df_local["layer_idx"] = layer_df_local["param"].str.extract(
            r"encoder\.layer\.(\d+)\.").astype(float)

        per_layer_stats = (layer_df_local.dropna(subset=["layer_idx"])
                           .groupby("layer_idx")["rel_frob"]
                           .agg(["mean", "std"])
                           .sort_index())
        per_layer = per_layer_stats["mean"]
        if not per_layer.empty:
            fig, ax = plt.subplots(figsize=(7, 3.2))
            ax.bar(per_layer.index.astype(int), per_layer.values,
                   yerr=per_layer_stats["std"].fillna(0).values,
                   color="tab:purple", capsize=2, ecolor="black",
                   error_kw={"linewidth": 0.7})
            ax.set_xlabel("ProtBert encoder layer index")
            ax.set_ylabel("Mean relative Frobenius drift")
            fig.tight_layout()
            fig.savefig(out_dir / f"layer_drift_{variant}_{ckpt.stem}.png", dpi=160)
            plt.close(fig)

        del ft_encoder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    # ---- Persist tables ----

    if len(variants_seen) == 1:
        suffix = f"{side}_{next(iter(variants_seen))}"
    else:
        suffix = f"{side}_all"

    drift_df = pd.DataFrame(drift_rows)
    collapse_df = pd.DataFrame(collapse_rows)
    layer_df = pd.concat(layer_rows, ignore_index=True)

    drift_df.to_csv(out_dir / f"drift_per_receptor_{suffix}.csv", index=False)
    collapse_df.to_csv(out_dir / f"collapse_metrics_{suffix}.csv", index=False)
    layer_df.to_csv(out_dir / f"layer_drift_{suffix}.csv", index=False)
    if subfamily_drift_rows:
        sf_df = pd.concat(subfamily_drift_rows, ignore_index=True).round(4)
        sf_df.to_csv(out_dir / f"subfamily_centroid_drift_{suffix}.csv",
                     index=False)

    print(f"\nVariants analyzed: {sorted(variants_seen)}")
    print(f"Wrote tables + figures to {out_dir}")

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)

    print("\n" + "=" * 70)
    print("DRIFT — cosine distance per (variant, pooling, group)")
    print("=" * 70)
    drift_summary = (drift_df
                     .groupby(["variant", "pooling", "group"])["cos_dist"]
                     .agg(["mean", "std", "count"])
                     .round(4))
    print(drift_summary)

    print("\n" + "=" * 70)
    print("SEED VARIANCE — per-seed drift means and coefficient of variation")
    print("=" * 70)
    # Per-checkpoint (= per-seed) mean drift: collapses receptors, keeps seeds.
    per_seed = (drift_df
                .groupby(["variant", "pooling", "group", "checkpoint"])["cos_dist"]
                .mean()
                .reset_index())
    # Aggregate across seeds: mean of per-seed drift means, std across seeds, CV.
    seed_var = (per_seed
                .groupby(["variant", "pooling", "group"])["cos_dist"]
                .agg(seed_mean="mean", seed_std="std", n_seeds="count")
                .round(4))
    seed_var["cv"] = (seed_var["seed_std"] / seed_var["seed_mean"]).round(3)
    # Also record the min and max per-seed drift so outlier runs are visible.
    extremes = (per_seed
                .groupby(["variant", "pooling", "group"])["cos_dist"]
                .agg(seed_min="min", seed_max="max")
                .round(4))
    seed_var = seed_var.join(extremes)
    print(seed_var)
    seed_var.to_csv(out_dir / f"summary_seed_variance_{suffix}.csv")
    print("  (CV > ~0.3 indicates substantive optimization instability;")
    print("   CV > ~1.0 means the std exceeds the mean — outlier runs dominate.)")

    print("\n" + "=" * 70)
    print("COLLAPSE — effective rank per group (rows: variant × pooling)")
    print("=" * 70)
    eff_rank = (collapse_df
                .pivot_table(values="effective_rank",
                             index=["variant", "pooling"],
                             columns="group")
                .round(2))
    print(eff_rank)

    print("\n" + "=" * 70)
    print("COLLAPSE — mean pairwise cosine distance per group")
    print("=" * 70)
    mpcd = (collapse_df
            .pivot_table(values="mean_pairwise_cos_dist",
                         index=["variant", "pooling"],
                         columns="group")
            .round(4))
    print(mpcd)

    print("\n" + "=" * 70)
    print(f"LAYER DRIFT — peak-drift {cfg['label']} encoder layer per variant")
    print("=" * 70)
    ld = layer_df.copy()
    ld["layer_idx"] = ld["param"].str.extract(r"encoder\.layer\.(\d+)\.").astype(float)
    per_layer = (ld.dropna(subset=["layer_idx"])
                 .groupby(["variant", "layer_idx"])["rel_frob"]
                 .mean())
    for variant in sorted(per_layer.index.get_level_values(0).unique()):
        series = per_layer.loc[variant]
        peak_idx = int(series.idxmax())
        print(f"  {variant:<10} peak layer = {peak_idx:>2}  "
              f"(rel_frob={series.max():.4f}); "
              f"layer-0={series.iloc[0]:.4f}, "
              f"layer-{int(series.index.max())}={series.iloc[-1]:.4f}")

    # ---- Per-variant layer-drift summary plot (averaged across the 5 seeds) ----

    per_seed_layer = (ld.dropna(subset=["layer_idx"])
                      .groupby(["variant", "checkpoint", "layer_idx"])["rel_frob"]
                      .mean()
                      .reset_index())
    seed_layer_stats = (per_seed_layer
                        .groupby(["variant", "layer_idx"])["rel_frob"]
                        .agg(["mean", "std", "count"])
                        .round(4))
    seed_layer_stats.to_csv(
        out_dir / f"summary_layer_drift_seed_avg_{suffix}.csv")
    for variant in sorted(seed_layer_stats.index.get_level_values(0).unique()):
        block = seed_layer_stats.loc[variant].sort_index()
        if block.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 3.2))
        ax.bar(block.index.astype(int), block["mean"].values,
               yerr=block["std"].fillna(0).values,
               color="tab:purple", capsize=2, ecolor="black",
               error_kw={"linewidth": 0.7})
        ax.set_xlabel("ProtBert encoder layer index")
        ax.set_ylabel("Relative Frobenius drift")
        fig.tight_layout()
        fig.savefig(out_dir / f"layer_drift_seed_avg_{variant}.png", dpi=180)
        plt.close(fig)

    print("\n" + "=" * 70)
    print("LAYER DRIFT — block-level breakdown within each encoder layer")
    print("=" * 70)
    ld["block"] = ld["param"].str.extract(
        r"\.(attention\.self|attention\.output|intermediate|output)\."
    )
    # Per-seed block-level mean first (one value per variant x checkpoint x
    # block), then aggregate across seeds so the bar-plot error bars show
    # seed-to-seed variability rather than within-block parameter spread.

    per_seed_block = (ld.dropna(subset=["block"])
                      .groupby(["variant", "checkpoint", "block"])["rel_frob"]
                      .mean()
                      .reset_index())
    block_stats = (per_seed_block
                   .groupby(["variant", "block"])["rel_frob"]
                   .agg(["mean", "std"])
                   .round(4))
    block_mean = block_stats["mean"].unstack("block")
    block_std = block_stats["std"].unstack("block").reindex_like(block_mean)
    print(block_mean)
    block_mean.to_csv(out_dir / f"summary_block_drift_{suffix}.csv")
    block_std.to_csv(out_dir / f"summary_block_drift_std_{suffix}.csv")

    # Keep `block_summary` so the headline-summary code further down still
    # finds the same name.
    block_summary = block_mean

    if not block_summary.empty:
        fig, ax = plt.subplots(figsize=(7, 3.6))
        block_mean.plot(kind="bar", ax=ax, yerr=block_std,
                        capsize=2, error_kw={"linewidth": 0.7,
                                             "ecolor": "black"})
        ax.set_ylabel("Mean relative Frobenius drift")
        ax.set_xlabel("Variant")
        ax.legend(title="block", bbox_to_anchor=(1.02, 1), loc="upper left",
                  frameon=True, fontsize=8)
        plt.setp(ax.get_xticklabels(), rotation=0)
        fig.tight_layout()
        fig.savefig(out_dir / f"block_drift_{suffix}.png", dpi=160)
        plt.close(fig)

    # Persist the same summaries as CSVs so they're easy to drop into a paper.
    drift_summary.to_csv(out_dir / f"summary_drift_{suffix}.csv")
    eff_rank.to_csv(out_dir / f"summary_effective_rank_{suffix}.csv")
    mpcd.to_csv(out_dir / f"summary_pairwise_cos_dist_{suffix}.csv")

    fine_variants = [v for v in sorted(variants_seen) if v != "pretrained"]
    columns = ["Pretrained"] + fine_variants

    # Drift mean / std per (variant, group), mean-pooling only.
    drift_grp = (drift_df[drift_df["pooling"] == "mean"]
                 .groupby(["variant", "group"])["cos_dist"]
                 .agg(["mean", "std"])
                 .round(2))
    # Effective rank and MPCD on the training group, aggregated across seeds.
    train_coll = collapse_df[(collapse_df["pooling"] == "mean")
                             & (collapse_df["group"] == "train")]
    er_stats = (train_coll.groupby("variant")["effective_rank"]
                .agg(["mean", "std", "min"]).round(2))
    mpcd_stats = (train_coll.groupby("variant")["mean_pairwise_cos_dist"]
                  .agg(["mean", "std"]).round(2))
    # Peak-drift layer per variant (per_layer was computed just above).
    peak_layer = {
        v: int(per_layer.loc[v].idxmax())
        for v in per_layer.index.get_level_values(0).unique()
    }
    # attention.output mean Frobenius drift per variant (from block_summary).
    attn_out_drift = (block_summary["attention.output"]
                      if "attention.output" in block_summary.columns
                      else pd.Series(dtype=float))

    def _fmt_drift(variant, group):
        try:
            row = drift_grp.loc[(variant, group)]
            return f"{row['mean']:.2f} ± {row['std']:.2f}"
        except KeyError:
            return "—"

    def _fmt_er(variant):
        if variant not in er_stats.index:
            return "—"
        r = er_stats.loc[variant]
        if variant == "pretrained":
            return f"{r['mean']:.2f}"
        return f"{r['mean']:.2f} ± {r['std']:.2f}"

    def _fmt_er_min(variant):
        if variant not in er_stats.index:
            return "—"
        return f"{er_stats.loc[variant, 'min']:.2f}"

    def _fmt_mpcd(variant):
        if variant not in mpcd_stats.index:
            return "—"
        r = mpcd_stats.loc[variant]
        if variant == "pretrained":
            return f"{r['mean']:.2f}"
        return f"{r['mean']:.2f} ± {r['std']:.2f}"

    def _fmt_attn(variant):
        return ("—" if variant not in attn_out_drift.index
                else f"{attn_out_drift.loc[variant]:.4f}")

    def _fmt_peak(variant):
        return str(peak_layer.get(variant, "—"))

    headline = pd.DataFrame(index=[
        "Per-receptor cosine drift, training receptors",
        "Per-receptor cosine drift, unseen-protein receptors",
        "Per-receptor cosine drift, unseen-ligand receptors",
        "Effective rank (training, mean ± SD across seeds)",
        "Effective rank (training, worst seed)",
        "Mean pairwise cosine distance (training)",
        "Mean Frobenius drift, attention.output block",
        "Peak-drift ProtBert layer index",
    ])
    headline.index.name = "Metric"

    for col_name, variant in zip(columns,
                                 ["pretrained"] + fine_variants):
        headline[col_name] = [
            _fmt_drift(variant, "train"),
            _fmt_drift(variant, "unseen_protein"),
            _fmt_drift(variant, "unseen_ligand"),
            _fmt_er(variant),
            _fmt_er_min(variant),
            _fmt_mpcd(variant),
            _fmt_attn(variant),
            _fmt_peak(variant),
        ]

    headline.to_csv(out_dir / f"summary_reviewer22_headline_{suffix}.csv")
    print("\n" + "=" * 70)
    print("REVIEWER 2.2 HEADLINE — drift / collapse / layer-drift summary")
    print("=" * 70)
    print(headline.to_string())
    print(f"\n  Saved to summary_reviewer22_headline_{suffix}.csv")

    print(f"\nSummary tables also written as summary_*_{suffix}.csv in {out_dir}")


def main(args):
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available()
                          and config["device"]["use_cuda"] else "cpu")
    print(f"Using device: {device}")

    out_dir = Path(config["output"]["results_dir"]) / "representation"
    out_dir.mkdir(parents=True, exist_ok=True)

    analyze_side(args, config, device, out_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Protein-side representation analysis of fine-tuned "
                    "ProtBert checkpoints (molecule side: see "
                    "scripts/molecule_representation_analysis.py)"
    )
    p.add_argument("--config", type=str, default="config.yaml")
    p.add_argument("--results_dir", type=str, default=None,
                   help="Directory to search for checkpoints (default: config output_dir)")
    p.add_argument("--glob", type=str, default=None,
                   help="Override the checkpoint glob. Defaults to "
                        "'cross_attention_prot_train_*_run*.pth'.")
    p.add_argument("--variant_name", type=str, default=None,
                   help="Override the variant label applied to every checkpoint. "
                        "If omitted, the variant is auto-inferred from each "
                        "checkpoint filename (CA-Prot, CA-Full, CA-Lig, CA-Base).")
    p.add_argument("--protein_model", type=str, default="Rostlab/prot_bert")
    p.add_argument("--max_len", type=int, default=None,
                   help="Tokenizer max length. Defaults to 1024.")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--pooling", type=str, default="mean",
                   choices=["mean", "cls", "both"],
                   help="Which residue pooling to use for receptor embeddings. "
                        "Default 'mean'. Use 'cls' for the [CLS] token only, or "
                        "'both' for the side-by-side report.")
    args = p.parse_args()
    main(args)
