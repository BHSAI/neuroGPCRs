"""Training script for XGBoost DTI model."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import pandas as pd
import yaml
import argparse
import xgboost as xgb
from pathlib import Path

from src.models.xgb_model import DTIFeatureExtractor
from src.utils.data_loader import load_datasets, extract_features, DTIDataset, create_dataloaders
from src.utils.metrics import evaluate_predictions, print_metrics, report_mean_std


def extract_features_for_xgb(model, dataloader, device):
    """Extract concatenated projected features from all batches."""
    model.eval()
    all_features, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            mol_embeddings = batch['mol_embeddings'].to(device)
            protein_embeddings = batch['protein_embeddings'].to(device)
            labels = batch['label']
            features = model(mol_embeddings, protein_embeddings)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.vstack(all_features), np.concatenate(all_labels)


def main(config_path: str = "config.yaml", protein_feat: str = None, mol_feat: str = None):
    """Main training function."""

    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and config['device']['use_cuda'] else "cpu")
    print(f"Using device: {device}")

    # Load datasets
    print("\nLoading datasets...")
    train_df, val_df, test_unseen_prot_df, test_unseen_lig_df = load_datasets(
        data_dir=config['data']['data_dir'],
        train_file=config['data']['train_file'],
        val_file=config['data']['val_file'],
        test_unseen_prot_file=config['data']['test_unseen_protein'],
        test_unseen_lig_file=config['data']['test_unseen_ligand']
    )

    # Check/generate feature files
    print("\nChecking feature files...")
    protein_feat_path = Path(protein_feat)
    mol_feat_path = Path(mol_feat)

    if not protein_feat_path.exists() or not mol_feat_path.exists():
        print("Feature files not found. Running embedding generation...")
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from generate_embeddings import generate_protein_embeddings, generate_molecule_embeddings

        all_dfs = [train_df, val_df, test_unseen_prot_df, test_unseen_lig_df]
        all_sequences = sorted(set(seq for df in all_dfs for seq in df['Target Sequence'].dropna().unique()))
        all_smiles = sorted(set(smi for df in all_dfs for smi in df['SMILES'].dropna().unique()))

        if not protein_feat_path.exists():
            print(f"  Generating protein embeddings -> {protein_feat}")
            generate_protein_embeddings(all_sequences, output_file=protein_feat)

        if not mol_feat_path.exists():
            print(f"  Generating molecule embeddings -> {mol_feat}")
            generate_molecule_embeddings(all_smiles, output_file=mol_feat)

    print("\nLoading pre-computed features...")
    prot_feat, mol_feat, target_dim, drug_dim = extract_features(
        prot_feat_file=protein_feat,
        mol_feat_file=mol_feat,
        protein_encoder=config['model']['protein_encoder'],
        molecule_encoder=config['model']['molecule_encoder']
    )

    # Create datasets and dataloaders (shared across seeds)
    train_dataset = DTIDataset(train_df, prot_feat, mol_feat)
    val_dataset = DTIDataset(val_df, prot_feat, mol_feat)
    test_unseen_prot_dataset = DTIDataset(test_unseen_prot_df, prot_feat, mol_feat)
    test_unseen_lig_dataset = DTIDataset(test_unseen_lig_df, prot_feat, mol_feat)

    train_loader, val_loader, test_unseen_prot_loader, test_unseen_lig_loader = create_dataloaders(
        train_dataset, val_dataset, test_unseen_prot_dataset, test_unseen_lig_dataset,
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers']
    )

    results_dir = Path(config['output']['results_dir'])
    results_dir.mkdir(exist_ok=True)

    # XGBoost hyperparameters
    xgb_cfg = config.get('xgboost', {})

    # Multi-seed training
    seeds = config['training'].get('seeds', [42, 123, 456, 789, 1024])
    all_metrics = {'Validation': [], 'Test (Unseen Protein)': [], 'Test (Unseen Ligand)': []}

    for run_idx, seed in enumerate(seeds, 1):
        print(f"\n--- Run {run_idx}/{len(seeds)} (seed {seed}) ---")
        torch.manual_seed(seed)

        # Initialize feature extractor (random projection, not trained)
        feature_extractor = DTIFeatureExtractor(
            drug_dim=drug_dim,
            target_dim=target_dim,
            latent_dim=config['model']['latent_dim']
        ).to(device)

        # Extract projected features
        print("Extracting features...")
        train_features, train_labels = extract_features_for_xgb(feature_extractor, train_loader, device)
        val_features, val_labels = extract_features_for_xgb(feature_extractor, val_loader, device)
        test_prot_features, test_prot_labels = extract_features_for_xgb(feature_extractor, test_unseen_prot_loader, device)
        test_lig_features, test_lig_labels = extract_features_for_xgb(feature_extractor, test_unseen_lig_loader, device)

        # Train XGBoost
        print("Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=xgb_cfg.get('n_estimators', 100),
            max_depth=xgb_cfg.get('max_depth', 6),
            learning_rate=xgb_cfg.get('learning_rate', 0.3),
            eval_metric='logloss',
            random_state=seed
        )

        xgb_model.fit(
            train_features, train_labels,
            eval_set=[(val_features, val_labels)],
            verbose=False
        )

        xgb_model.save_model(str(results_dir / f"xgb_run{run_idx}.json"))

        # Evaluate — Validation
        val_probs = xgb_model.predict_proba(val_features)[:, 1]
        val_df['Predictions'] = (val_probs > 0.5).astype(int)
        val_df['Predictions_Proba'] = val_probs
        val_df.to_csv(results_dir / f"val_predictions_xgb_run{run_idx}.csv")
        val_metrics = evaluate_predictions(val_df)
        all_metrics['Validation'].append(val_metrics)
        print_metrics(val_metrics, "Validation")

        # Evaluate — Test (unseen protein)
        test_prot_probs = xgb_model.predict_proba(test_prot_features)[:, 1]
        test_unseen_prot_df['Predictions'] = (test_prot_probs > 0.5).astype(int)
        test_unseen_prot_df['Predictions_Proba'] = test_prot_probs
        test_unseen_prot_df.to_csv(results_dir / f"test_unseen_protein_xgb_run{run_idx}.csv")
        test_prot_metrics = evaluate_predictions(test_unseen_prot_df)
        all_metrics['Test (Unseen Protein)'].append(test_prot_metrics)
        print_metrics(test_prot_metrics, "Test (Unseen Protein)")

        # Evaluate — Test (unseen ligand)
        test_lig_probs = xgb_model.predict_proba(test_lig_features)[:, 1]
        test_unseen_lig_df['Predictions'] = (test_lig_probs > 0.5).astype(int)
        test_unseen_lig_df['Predictions_Proba'] = test_lig_probs
        test_unseen_lig_df.to_csv(results_dir / f"test_unseen_ligand_xgb_run{run_idx}.csv")
        test_lig_metrics = evaluate_predictions(test_unseen_lig_df)
        all_metrics['Test (Unseen Ligand)'].append(test_lig_metrics)
        print_metrics(test_lig_metrics, "Test (Unseen Ligand)")

    # Report mean ± std across all seeds
    report_mean_std(all_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost DTI model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--protein_feat", type=str, default="ProtBert_features.h5", help="Path to protein features h5 file")
    parser.add_argument("--mol_feat", type=str, default="MolFormer_features.h5", help="Path to molecule features h5 file")

    args = parser.parse_args()
    main(args.config, args.protein_feat, args.mol_feat)
