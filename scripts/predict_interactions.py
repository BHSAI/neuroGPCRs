"""
Inference script for predicting GPCR-ligand interactions.

Given SMILES string(s), predict binding probabilities against all GPCRs
from the training set.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import pandas as pd
import yaml
import argparse
from pathlib import Path
from transformers import AutoTokenizer
from tqdm import tqdm

from src.models.cross_attention_finetune import DTIFineTuneCrossAttention


def load_gpcrs_from_training(data_path: str = "data/training_set.csv"):
    """
    Load unique GPCRs from training set.

    Returns:
        DataFrame with UniProt IDs and protein sequences
    """
    df = pd.read_csv(data_path, index_col=0)

    # Get unique GPCR proteins
    gpcrs = df[['UniProt', 'Target Sequence']].drop_duplicates()
    gpcrs = gpcrs.reset_index(drop=True)

    print(f"Loaded {len(gpcrs)} unique GPCRs from training set")

    return gpcrs


def predict_interactions(
    smiles_list,
    gpcrs_df,
    model_path,
    protein_model_name="Rostlab/prot_bert",
    molecule_model_name="ibm/MolFormer-XL-both-10pct",
    d_model=512,
    n_heads=4,
    dropout=0.1,
    device=None,
    batch_size=16,
    max_protein_len=1024,
    max_molecule_len=512
):
    """
    Predict binding probabilities for SMILES against all GPCRs.

    Args:
        smiles_list: List of SMILES strings
        gpcrs_df: DataFrame with UniProt and Target Sequence columns
        model_path: Path to trained model weights
        protein_model_name: HuggingFace protein model name
        molecule_model_name: HuggingFace molecule model name
        d_model: Model dimensionality
        n_heads: Number of attention heads
        dropout: Dropout rate
        device: Device to run on (auto-detected if None)
        batch_size: Batch size for inference
        max_protein_len: Maximum protein sequence length
        max_molecule_len: Maximum SMILES length

    Returns:
        DataFrame with predictions
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nUsing device: {device}")

    # Load tokenizers
    print("Loading tokenizers...")
    protein_tokenizer = AutoTokenizer.from_pretrained(protein_model_name, do_lower_case=False)
    molecule_tokenizer = AutoTokenizer.from_pretrained(molecule_model_name, trust_remote_code=True)

    # Load model
    print(f"Loading model from {model_path}...")
    model = DTIFineTuneCrossAttention(
        protein_model_name=protein_model_name,
        molecule_model_name=molecule_model_name,
        d_model=d_model,
        n_heads=n_heads,
        dropout=dropout,
        freeze_encoders=True  # For inference
    ).to(device)

    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"Model loaded successfully!")

    # Prepare all protein-SMILES pairs
    results = []

    print(f"\nPredicting interactions for {len(smiles_list)} compound(s) against {len(gpcrs_df)} GPCRs...")

    for smiles in smiles_list:
        print(f"\nProcessing: {smiles}")

        # Prepare batches
        batch_proteins = []
        batch_smiles = []
        batch_uniprots = []

        for idx, row in gpcrs_df.iterrows():
            batch_uniprots.append(row['UniProt'])
            batch_proteins.append(row['Target Sequence'])
            batch_smiles.append(smiles)

            # Process batch
            if len(batch_proteins) == batch_size or idx == len(gpcrs_df) - 1:
                # Tokenize proteins
                spaced_proteins = [' '.join(list(seq)) for seq in batch_proteins]
                protein_tokens = protein_tokenizer(
                    spaced_proteins,
                    max_length=max_protein_len,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )

                # Tokenize molecules
                molecule_tokens = molecule_tokenizer(
                    batch_smiles,
                    max_length=max_molecule_len,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )

                # Move to device
                protein_input_ids = protein_tokens['input_ids'].to(device)
                protein_attention_mask = protein_tokens['attention_mask'].to(device)
                molecule_input_ids = molecule_tokens['input_ids'].to(device)
                molecule_attention_mask = molecule_tokens['attention_mask'].to(device)

                # Predict
                with torch.no_grad():
                    predictions = model(
                        protein_input_ids=protein_input_ids,
                        protein_attention_mask=protein_attention_mask,
                        molecule_input_ids=molecule_input_ids,
                        molecule_attention_mask=molecule_attention_mask
                    )

                # Store results
                for i, (uniprot, protein, smi) in enumerate(zip(batch_uniprots, batch_proteins, batch_smiles)):
                    results.append({
                        'SMILES': smi,
                        'UniProt': uniprot,
                        'Target_Sequence': protein,
                        'Binding_Probability': predictions[i].item()
                    })

                # Clear batch
                batch_proteins = []
                batch_smiles = []
                batch_uniprots = []

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Sort by binding probability (descending)
    results_df = results_df.sort_values('Binding_Probability', ascending=False)

    print(f"\nPredictions complete!")

    return results_df


def main():
    parser = argparse.ArgumentParser(
        description="Predict GPCR-ligand interactions for given SMILES"
    )
    parser.add_argument(
        "--smiles",
        type=str,
        help="Single SMILES string"
    )
    parser.add_argument(
        "--smiles_file",
        type=str,
        help="File containing SMILES strings (one per line)"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model (.pth file)"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/training_set.csv",
        help="Path to training data CSV (to extract GPCRs)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions.csv",
        help="Output CSV file for predictions"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Only save top K predictions per compound"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Only save predictions above this threshold"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file (d_model, n_heads, dropout read from finetune section)"
    )

    args = parser.parse_args()

    # Get SMILES list
    smiles_list = []
    if args.smiles:
        smiles_list.append(args.smiles)
    elif args.smiles_file:
        with open(args.smiles_file, 'r') as f:
            smiles_list = [line.strip() for line in f if line.strip()]
    else:
        print("Error: Please provide either --smiles or --smiles_file")
        return

    # Load config for model architecture params
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print(f"\nGPCR-Ligand Binding Prediction")
    print(f"  Compounds: {len(smiles_list)}")
    print(f"  Model: {args.model}")
    print(f"  Config: {args.config}")
    print(f"  GPCR data: {args.data}")

    # Load GPCRs
    gpcrs_df = load_gpcrs_from_training(args.data)

    # Make predictions
    results_df = predict_interactions(
        smiles_list=smiles_list,
        gpcrs_df=gpcrs_df,
        model_path=args.model,
        batch_size=args.batch_size,
        d_model=config['finetune']['d_model'],
        n_heads=config['finetune']['n_heads'],
        dropout=config['finetune']['dropout']
    )

    # Apply filters if specified
    if args.threshold is not None:
        print(f"\nFiltering predictions with probability >= {args.threshold}")
        results_df = results_df[results_df['Binding_Probability'] >= args.threshold]

    if args.top_k is not None:
        print(f"\nKeeping top {args.top_k} predictions per compound")
        results_df = results_df.groupby('SMILES').head(args.top_k)

    # Save results
    results_df.to_csv(args.output, index=False)
    print(f"\nResults saved to: {args.output}")
    print(f"Total predictions: {len(results_df)}")

    # Show top predictions
    print("\nTop 10 Predictions:")
    print(results_df[['SMILES', 'UniProt', 'Binding_Probability']].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
