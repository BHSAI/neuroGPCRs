"""Data loading and preprocessing utilities for DTI prediction."""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
from pathlib import Path
from typing import Tuple, Dict, Optional, Callable


class DTIDataset(Dataset):
    """Dataset class for Drug-Target Interaction data with pre-computed embeddings."""

    def __init__(self, dataframe: pd.DataFrame, prot_feat: h5py.File, mol_feat: h5py.File):
        """
        Initialize DTI dataset.

        Args:
            dataframe: DataFrame containing protein sequences, SMILES, and labels
            prot_feat: h5py file containing protein embeddings
            mol_feat: h5py file containing molecular embeddings
        """
        self.proteins = dataframe['Target Sequence'].tolist()
        self.smiles = dataframe['SMILES'].tolist()
        self.labels = dataframe['Label'].tolist()
        self.prot_feat = prot_feat
        self.mol_feat = mol_feat

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        protein_seq = self.proteins[idx]
        smiles_str = self.smiles[idx]
        label = self.labels[idx]

        # Extract embeddings from h5 files
        protein_inputs = np.squeeze(self.prot_feat[protein_seq][()])
        mol_inputs = np.squeeze(self.mol_feat[smiles_str][()])

        return {
            'protein_embeddings': torch.tensor(protein_inputs, dtype=torch.float32),
            'mol_embeddings': torch.tensor(mol_inputs, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.float32)
        }


def load_datasets(
    data_dir: str = "data",
    train_file: str = "training_set.csv",
    val_file: str = "validation_set.csv",
    test_unseen_prot_file: str = "test_set_unseen_protein.csv",
    test_unseen_lig_file: str = "test_set_unseen_ligands.csv"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all datasets from CSV files.

    Args:
        data_dir: Directory containing data files
        train_file: Training set filename
        val_file: Validation set filename
        test_unseen_prot_file: Test set with unseen proteins filename
        test_unseen_lig_file: Test set with unseen ligands filename

    Returns:
        Tuple of (train_df, val_df, test_unseen_prot_df, test_unseen_lig_df)
    """
    data_path = Path(data_dir)

    train_df = pd.read_csv(data_path / train_file, index_col=0).dropna().reset_index(drop=True)
    val_df = pd.read_csv(data_path / val_file, index_col=0).dropna().reset_index(drop=True)
    test_unseen_prot_df = pd.read_csv(data_path / test_unseen_prot_file, index_col=0).dropna().reset_index(drop=True)
    test_unseen_lig_df = pd.read_csv(data_path / test_unseen_lig_file, index_col=0).dropna().reset_index(drop=True)

    print(f"Loaded datasets:")
    print(f"  Training: {len(train_df)} samples")
    print(f"  Validation: {len(val_df)} samples")
    print(f"  Test (unseen protein): {len(test_unseen_prot_df)} samples")
    print(f"  Test (unseen ligand): {len(test_unseen_lig_df)} samples")

    return train_df, val_df, test_unseen_prot_df, test_unseen_lig_df


def extract_features(
    prot_feat_file: str,
    mol_feat_file: str,
    protein_encoder: str = "ProtBert",
    molecule_encoder: str = "MolFormer"
) -> Tuple[h5py.File, h5py.File, int, int]:
    """
    Load feature files and determine embedding dimensions.

    Args:
        prot_feat_file: Path to protein features h5 file
        mol_feat_file: Path to molecule features h5 file
        protein_encoder: Name of protein encoder (ProtBert or ESM)
        molecule_encoder: Name of molecule encoder (MolFormer or ChemBERTa)

    Returns:
        Tuple of (prot_feat, mol_feat, target_dim, drug_dim)
    """
    prot_feat = h5py.File(prot_feat_file, 'r')
    mol_feat = h5py.File(mol_feat_file, 'r')

    # Defining target and drug dimensions from ProtBERT and MolFormer

    target_dim = 1024
    drug_dim = 768

    print(f"Using {protein_encoder} (dim={target_dim}) and {molecule_encoder} (dim={drug_dim})")

    return prot_feat, mol_feat, target_dim, drug_dim


def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_unseen_prot_dataset: Dataset,
    test_unseen_lig_dataset: Dataset,
    batch_size: int = 32,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoader objects for all datasets.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_unseen_prot_dataset: Test dataset with unseen proteins
        test_unseen_lig_dataset: Test dataset with unseen ligands
        batch_size: Batch size for training
        num_workers: Number of workers for data loading

    Returns:
        Tuple of (train_loader, val_loader, test_unseen_prot_loader, test_unseen_lig_loader)
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_unseen_prot_loader = DataLoader(test_unseen_prot_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_unseen_lig_loader = DataLoader(test_unseen_lig_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_unseen_prot_loader, test_unseen_lig_loader


class DTIFineTuneDataset(Dataset):
    """Dataset class for DTI with raw sequences and SMILES (for cross-attention fine-tuning)."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        protein_tokenizer: Callable,
        molecule_tokenizer: Callable,
        max_protein_len: int = 1024,
        max_molecule_len: int = 512
    ):
        """
        Initialize fine-tuning dataset.

        Args:
            dataframe: DataFrame containing protein sequences, SMILES, and labels
            protein_tokenizer: Tokenizer for protein sequences
            molecule_tokenizer: Tokenizer for molecules
            max_protein_len: Maximum protein sequence length
            max_molecule_len: Maximum molecule sequence length
        """
        self.proteins = dataframe['Target Sequence'].tolist()
        self.smiles = dataframe['SMILES'].tolist()
        self.labels = dataframe['Label'].tolist()
        self.protein_tokenizer = protein_tokenizer
        self.molecule_tokenizer = molecule_tokenizer
        self.max_protein_len = max_protein_len
        self.max_molecule_len = max_molecule_len

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        protein_seq = self.proteins[idx]
        smiles_str = self.smiles[idx]
        label = self.labels[idx]

        # Tokenize protein (add spaces between amino acids for ProtBert)
        spaced_protein = ' '.join(list(protein_seq))
        protein_tokens = self.protein_tokenizer(
            spaced_protein,
            max_length=self.max_protein_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Tokenize molecule
        molecule_tokens = self.molecule_tokenizer(
            smiles_str,
            max_length=self.max_molecule_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'protein_input_ids': protein_tokens['input_ids'].squeeze(0),
            'protein_attention_mask': protein_tokens['attention_mask'].squeeze(0),
            'molecule_input_ids': molecule_tokens['input_ids'].squeeze(0),
            'molecule_attention_mask': molecule_tokens['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.float32)
        }
