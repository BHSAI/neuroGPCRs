"""
Generate and save pre-computed embeddings for proteins and molecules.

This script generates embeddings using ProtBert (for proteins) and MolFormer
(for molecules) from the dataset CSV files. Embeddings are saved as HDF5 files
for efficient loading during training.
"""

import torch
import h5py
import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


def generate_protein_embeddings(
    sequences,
    model_name="Rostlab/prot_bert",
    output_file="ProtBert_features.h5",
    batch_size=8,
    max_length=1024,
    device=None
):
    """
    Generate protein embeddings using ProtBert.

    Args:
        sequences: List of unique protein sequences
        model_name: HuggingFace model name for protein encoder
        output_file: Output HDF5 file path
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        device: Device to run on (auto-detected if None)

    Returns:
        Path to saved HDF5 file
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nGenerating Protein Embeddings with {model_name}")
    print(f"  Device: {device}")
    print(f"  Unique sequences: {len(sequences)}")
    print(f"  Output file: {output_file}")

    # Load model and tokenizer
    print("\nLoading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
    model = AutoModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    print(f"Model loaded: {model_name}")
    print(f"Hidden size: {model.config.hidden_size}")

    # Create HDF5 file
    h5_file = h5py.File(output_file, 'w')

    # Process sequences in batches
    print("\nGenerating embeddings...")
    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size)):
            batch_sequences = sequences[i:i + batch_size]

            # Add spaces between amino acids (required for ProtBert)
            spaced_sequences = [' '.join(list(seq)) for seq in batch_sequences]

            # Tokenize
            inputs = tokenizer(
                spaced_sequences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate embeddings
            outputs = model(**inputs)

            # Mean pooling over sequence dimension
            # Shape: (batch_size, seq_len, hidden_size) -> (batch_size, hidden_size)
            attention_mask = inputs['attention_mask'].unsqueeze(-1)
            embeddings = (outputs.last_hidden_state * attention_mask).sum(1) / attention_mask.sum(1)
            embeddings = embeddings.cpu().numpy()

            # Save to HDF5
            for seq, emb in zip(batch_sequences, embeddings):
                h5_file.create_dataset(seq, data=emb)

    h5_file.close()
    print(f"\n Protein embeddings saved to: {output_file}")
    print(f"  File size: {Path(output_file).stat().st_size / 1024**2:.2f} MB")

    return output_file


def generate_molecule_embeddings(
    smiles_list,
    model_name="ibm/MolFormer-XL-both-10pct",
    output_file="MolFormer_features.h5",
    batch_size=32,
    max_length=512,
    device=None
):
    """
    Generate molecule embeddings using MolFormer.

    Args:
        smiles_list: List of unique SMILES strings
        model_name: HuggingFace model name for molecule encoder
        output_file: Output HDF5 file path
        batch_size: Batch size for processing
        max_length: Maximum SMILES length
        device: Device to run on (auto-detected if None)

    Returns:
        Path to saved HDF5 file
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nGenerating Molecule Embeddings with {model_name}")
    print(f"  Device: {device}")
    print(f"  Unique SMILES: {len(smiles_list)}")
    print(f"  Output file: {output_file}")

    # Load model and tokenizer
    print("\nLoading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model = model.to(device)
    model.eval()

    print(f"Model loaded: {model_name}")
    print(f"Hidden size: {model.config.hidden_size}")

    # Create HDF5 file
    h5_file = h5py.File(output_file, 'w')

    # Process SMILES in batches
    print("\nGenerating embeddings...")
    with torch.no_grad():
        for i in tqdm(range(0, len(smiles_list), batch_size)):
            batch_smiles = smiles_list[i:i + batch_size]

            # Tokenize
            inputs = tokenizer(
                batch_smiles,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate embeddings
            outputs = model(**inputs)

            # Mean pooling over sequence dimension
            # Shape: (batch_size, seq_len, hidden_size) -> (batch_size, hidden_size)
            attention_mask = inputs['attention_mask'].unsqueeze(-1)
            embeddings = (outputs.last_hidden_state * attention_mask).sum(1) / attention_mask.sum(1)
            embeddings = embeddings.cpu().numpy()

            # Save to HDF5
            for smiles, emb in zip(batch_smiles, embeddings):
                h5_file.create_dataset(smiles, data=emb)

    h5_file.close()
    print(f"\n Molecule embeddings saved to: {output_file}")
    print(f"  File size: {Path(output_file).stat().st_size / 1024**2:.2f} MB")

    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Generate pre-computed embeddings for proteins and molecules"
    )
    parser.add_argument(
        "--data_files",
        nargs="+",
        default=["data/training_set.csv", "data/validation_set.csv",
                 "data/test_set_unseen_protein.csv", "data/test_set_unseen_ligands.csv"],
        help="CSV files to extract sequences/SMILES from"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory to save embedding files"
    )
    parser.add_argument(
        "--protein_model",
        type=str,
        default="Rostlab/prot_bert",
        help="HuggingFace protein model name"
    )
    parser.add_argument(
        "--molecule_model",
        type=str,
        default="ibm/MolFormer-XL-both-10pct",
        help="HuggingFace molecule model name"
    )
    parser.add_argument(
        "--protein_batch_size",
        type=int,
        default=8,
        help="Batch size for protein encoding"
    )
    parser.add_argument(
        "--molecule_batch_size",
        type=int,
        default=32,
        help="Batch size for molecule encoding"
    )
    parser.add_argument(
        "--skip_proteins",
        action="store_true",
        help="Skip protein embedding generation"
    )
    parser.add_argument(
        "--skip_molecules",
        action="store_true",
        help="Skip molecule embedding generation"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Collect unique sequences and SMILES from all data files
    print("\nLoading data files...")

    all_sequences = set()
    all_smiles = set()

    for file_path in args.data_files:
        if not Path(file_path).exists():
            print(f"Warning: File not found: {file_path}")
            continue

        print(f"Reading: {file_path}")
        df = pd.read_csv(file_path, index_col=0)

        sequences = df['Target Sequence'].dropna().unique()
        smiles = df['SMILES'].dropna().unique()

        all_sequences.update(sequences)
        all_smiles.update(smiles)

        print(f"  Sequences: {len(sequences)}, SMILES: {len(smiles)}")

    print(f"\nTotal unique sequences: {len(all_sequences)}")
    print(f"Total unique SMILES: {len(all_smiles)}")

    # Convert to sorted lists for reproducibility
    all_sequences = sorted(list(all_sequences))
    all_smiles = sorted(list(all_smiles))

    # Generate protein embeddings
    if not args.skip_proteins:
        protein_output = output_dir / "ProtBert_features.h5"
        generate_protein_embeddings(
            sequences=all_sequences,
            model_name=args.protein_model,
            output_file=str(protein_output),
            batch_size=args.protein_batch_size
        )
    else:
        print("\nSkipping protein embedding generation")

    # Generate molecule embeddings
    if not args.skip_molecules:
        molecule_output = output_dir / "MolFormer_features.h5"
        generate_molecule_embeddings(
            smiles_list=all_smiles,
            model_name=args.molecule_model,
            output_file=str(molecule_output),
            batch_size=args.molecule_batch_size
        )
    else:
        print("\nSkipping molecule embedding generation")

    print("\nEmbedding generation complete.")


if __name__ == "__main__":
    main()
