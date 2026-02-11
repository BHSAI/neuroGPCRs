"""Unified training script for Cross-Attention DTI model with flexible fine-tuning options."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import yaml
import argparse
from pathlib import Path
from transformers import AutoTokenizer

from src.models.cross_attention_finetune import DTIFineTuneCrossAttention
from src.utils.data_loader import load_datasets, create_dataloaders, DTIFineTuneDataset
from src.utils.training import train_model_finetune, evaluate_model_finetune
from src.utils.metrics import print_metrics, evaluate_predictions, report_mean_std


def main(args):
    """Main training function with flexible freezing options."""

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    freeze_protein = args.freeze_protein
    freeze_molecule = args.freeze_molecule

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and config['device']['use_cuda'] else "cpu")
    print(f"Using device: {device}")

    # Load tokenizers (shared across seeds)
    print("\nLoading tokenizers...")
    protein_model_name = config['finetune']['protein_model']
    molecule_model_name = config['finetune']['molecule_model']

    protein_tokenizer = AutoTokenizer.from_pretrained(protein_model_name, do_lower_case=False)
    molecule_tokenizer = AutoTokenizer.from_pretrained(molecule_model_name, trust_remote_code=True)

    # Load datasets (shared across seeds)
    print("\nLoading datasets...")
    train_df, val_df, test_unseen_prot_df, test_unseen_lig_df = load_datasets(
        data_dir=config['data']['data_dir'],
        train_file=config['data']['train_file'],
        val_file=config['data']['val_file'],
        test_unseen_prot_file=config['data']['test_unseen_protein'],
        test_unseen_lig_file=config['data']['test_unseen_ligand']
    )

    # Create datasets and dataloaders (shared across seeds)
    train_dataset = DTIFineTuneDataset(
        train_df, protein_tokenizer, molecule_tokenizer,
        max_protein_len=config['finetune']['max_protein_len'],
        max_molecule_len=config['finetune']['max_molecule_len']
    )
    val_dataset = DTIFineTuneDataset(
        val_df, protein_tokenizer, molecule_tokenizer,
        max_protein_len=config['finetune']['max_protein_len'],
        max_molecule_len=config['finetune']['max_molecule_len']
    )
    test_unseen_prot_dataset = DTIFineTuneDataset(
        test_unseen_prot_df, protein_tokenizer, molecule_tokenizer,
        max_protein_len=config['finetune']['max_protein_len'],
        max_molecule_len=config['finetune']['max_molecule_len']
    )
    test_unseen_lig_dataset = DTIFineTuneDataset(
        test_unseen_lig_df, protein_tokenizer, molecule_tokenizer,
        max_protein_len=config['finetune']['max_protein_len'],
        max_molecule_len=config['finetune']['max_molecule_len']
    )

    train_loader, val_loader, test_unseen_prot_loader, test_unseen_lig_loader = create_dataloaders(
        train_dataset, val_dataset, test_unseen_prot_dataset, test_unseen_lig_dataset,
        batch_size=config['finetune']['batch_size'],
        num_workers=config['training']['num_workers']
    )

    results_dir = Path(config['output']['results_dir'])
    results_dir.mkdir(exist_ok=True)

    freeze_config = f"prot{'_frozen' if freeze_protein else '_train'}_mol{'_frozen' if freeze_molecule else '_train'}"

    # Multi-seed training
    seeds = config['training'].get('seeds', [42, 123, 456, 789, 1024])
    all_metrics = {'Validation': [], 'Test (Unseen Protein)': [], 'Test (Unseen Ligand)': []}

    for run_idx, seed in enumerate(seeds, 1):
        print(f"\n--- Run {run_idx}/{len(seeds)} (seed {seed}) ---")
        torch.manual_seed(seed)

        # Initialize model
        print(f"  Protein Encoder: {'FROZEN' if freeze_protein else 'TRAINABLE'}")
        print(f"  Molecule Encoder: {'FROZEN' if freeze_molecule else 'TRAINABLE'}")

        freeze_both = freeze_protein and freeze_molecule
        model = DTIFineTuneCrossAttention(
            protein_model_name=protein_model_name,
            molecule_model_name=molecule_model_name,
            d_model=config['finetune']['d_model'],
            n_heads=config['finetune']['n_heads'],
            dropout=config['finetune']['dropout'],
            freeze_encoders=freeze_both
        ).to(device)

        # Apply individual freezing if not both frozen
        if not freeze_both:
            if freeze_protein:
                for param in model.protein_encoder.parameters():
                    param.requires_grad = False
            if freeze_molecule:
                for param in model.molecule_encoder.parameters():
                    param.requires_grad = False

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Trainable parameters: {trainable_params:,}")

        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['finetune']['learning_rate'],
            weight_decay=config['finetune']['weight_decay']
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.1, patience=3, verbose=True
        )

        criterion = nn.BCELoss()

        # Train
        save_path = results_dir / f"cross_attention_{freeze_config}_run{run_idx}.pth"
        model, history = train_model_finetune(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            num_epochs=config['finetune']['num_epochs'],
            save_path=str(save_path),
            verbose=True,
            early_stopping_patience=config['finetune']['early_stopping_patience']
        )

        history_df = pd.DataFrame(history)
        history_df.to_csv(results_dir / f"history_{freeze_config}_run{run_idx}.csv", index=False)

        # Evaluate
        val_preds, val_probs, *_ = evaluate_model_finetune(model, val_loader, criterion, device, verbose=False)
        val_df['Predictions'] = val_preds
        val_df['Predictions_Proba'] = val_probs
        val_df.to_csv(results_dir / f"val_predictions_{freeze_config}_run{run_idx}.csv")
        val_metrics = evaluate_predictions(val_df)
        all_metrics['Validation'].append(val_metrics)
        print_metrics(val_metrics, "Validation")

        test_preds, test_probs, *_ = evaluate_model_finetune(model, test_unseen_prot_loader, criterion, device, verbose=False)
        test_unseen_prot_df['Predictions'] = test_preds
        test_unseen_prot_df['Predictions_Proba'] = test_probs
        test_unseen_prot_df.to_csv(results_dir / f"test_unseen_protein_{freeze_config}_run{run_idx}.csv")
        test_prot_metrics = evaluate_predictions(test_unseen_prot_df)
        all_metrics['Test (Unseen Protein)'].append(test_prot_metrics)
        print_metrics(test_prot_metrics, "Test (Unseen Protein)")

        test_preds, test_probs, *_ = evaluate_model_finetune(model, test_unseen_lig_loader, criterion, device, verbose=False)
        test_unseen_lig_df['Predictions'] = test_preds
        test_unseen_lig_df['Predictions_Proba'] = test_probs
        test_unseen_lig_df.to_csv(results_dir / f"test_unseen_ligand_{freeze_config}_run{run_idx}.csv")
        test_lig_metrics = evaluate_predictions(test_unseen_lig_df)
        all_metrics['Test (Unseen Ligand)'].append(test_lig_metrics)
        print_metrics(test_lig_metrics, "Test (Unseen Ligand)")

    # Report mean ± std across all seeds
    report_mean_std(all_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Cross-Attention DTI model with flexible encoder freezing options"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--freeze_protein",
        action="store_true",
        help="Freeze protein encoder weights (ProtBert)"
    )
    parser.add_argument(
        "--freeze_molecule",
        action="store_true",
        help="Freeze molecule encoder weights (MolFormer)"
    )

    args = parser.parse_args()

    print(f"\nCross-Attention Model Configuration:")
    print(f"  Freeze Protein Encoder: {args.freeze_protein}")
    print(f"  Freeze Molecule Encoder: {args.freeze_molecule}")

    main(args)
