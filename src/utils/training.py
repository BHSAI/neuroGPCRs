"""Training and evaluation utilities for DTI models."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score
from imblearn.metrics import specificity_score
from typing import Tuple, List
from tqdm import tqdm


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    verbose: bool = True
) -> Tuple[float, float, float, float, float, float, float]:
    """
    Train model for one epoch.

    Args:
        model: PyTorch model to train
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run on
        verbose: Whether to show progress bar

    Returns:
        Tuple of (loss, accuracy, precision, recall, specificity, mcc, auc)
    """
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    iterator = tqdm(dataloader, desc="Training") if verbose else dataloader

    for batch in iterator:
        mol_embeddings = batch['mol_embeddings'].to(device)
        protein_embeddings = batch['protein_embeddings'].to(device)
        labels = batch['label'].unsqueeze(1).to(device)

        optimizer.zero_grad()
        logits = torch.clamp(model(mol_embeddings, protein_embeddings), max=1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = (logits > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(logits.cpu().detach().numpy())

    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    specificity = specificity_score(all_labels, all_preds, average='binary', pos_label=1)

    try:
        auc = roc_auc_score(all_labels, all_probs)
        mcc = matthews_corrcoef(all_labels, all_preds)
    except ValueError:
        auc = 0.5
        mcc = 0
        print("Warning: Could not calculate AUC/MCC")

    return avg_loss, accuracy, precision, recall, specificity, mcc, auc


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    verbose: bool = True
) -> Tuple[List, List, float, float, float, float, float, float, float]:
    """
    Evaluate model on a dataset.

    Args:
        model: PyTorch model to evaluate
        dataloader: Evaluation data loader
        criterion: Loss function
        device: Device to run on
        verbose: Whether to show progress bar

    Returns:
        Tuple of (predictions, probabilities, loss, accuracy, precision, recall, specificity, mcc, auc)
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    iterator = tqdm(dataloader, desc="Evaluating") if verbose else dataloader

    with torch.no_grad():
        for batch in iterator:
            mol_embeddings = batch['mol_embeddings'].to(device)
            protein_embeddings = batch['protein_embeddings'].to(device)
            labels = batch['label'].unsqueeze(1).to(device)

            logits = torch.clamp(model(mol_embeddings, protein_embeddings), max=1)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = (logits > 0.5).float()
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
            all_probs.extend(logits.cpu().numpy().flatten())

    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    specificity = specificity_score(all_labels, all_preds, average='binary', pos_label=1)

    try:
        auc = roc_auc_score(all_labels, all_probs)
        mcc = matthews_corrcoef(all_labels, all_preds)
    except ValueError:
        auc = 0.5
        mcc = 0
        print("Warning: Could not calculate AUC/MCC")

    return all_preds, all_probs, avg_loss, accuracy, precision, recall, specificity, mcc, auc


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int = 50,
    save_path: str = "best_model.pth",
    early_stopping_patience: int = 5,
    verbose: bool = True
) -> Tuple[nn.Module, List[dict]]:
    """
    Complete training loop with validation and early stopping.

    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run on
        num_epochs: Number of training epochs
        save_path: Path to save best model
        early_stopping_patience: Number of epochs without val AUC improvement before stopping
        verbose: Whether to print progress

    Returns:
        Tuple of (best_model, history)
    """
    best_val_auc = 0
    patience_counter = 0
    history = []

    for epoch in range(num_epochs):
        if verbose:
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)

        # Training
        train_loss, train_acc, train_prec, train_rec, train_spec, train_mcc, train_auc = train_epoch(
            model, train_loader, criterion, optimizer, device, verbose=verbose
        )

        # Validation
        val_preds, val_probs, val_loss, val_acc, val_prec, val_rec, val_spec, val_mcc, val_auc = evaluate_model(
            model, val_loader, criterion, device, verbose=verbose
        )

        # Save history
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_auc': train_auc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_auc': val_auc,
            'val_mcc': val_mcc
        })

        if verbose:
            print(f"Train - Loss: {train_loss:.3f}, Acc: {train_acc:.3f}, AUC: {train_auc:.3f}, "
                  f"Sens: {train_rec:.3f}, Spec: {train_spec:.3f}, MCC: {train_mcc:.3f}")
            print(f"Val   - Loss: {val_loss:.3f}, Acc: {val_acc:.3f}, AUC: {val_auc:.3f}, "
                  f"Sens: {val_rec:.3f}, Spec: {val_spec:.3f}, MCC: {val_mcc:.3f}")

        # Save best model and early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
            if verbose:
                print(f"New best model saved! Val AUC: {best_val_auc:.4f}")
        else:
            patience_counter += 1
            if verbose:
                print(f"No improvement. Patience: {patience_counter}/{early_stopping_patience}")

        # Early stopping
        if patience_counter >= early_stopping_patience:
            if verbose:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break

    # Load best model
    model.load_state_dict(torch.load(save_path))

    return model, history


# ---------------------------------------------------------------------------
# Fine-tuning variants (cross-attention: tokenised inputs, gradient clipping,
# LR scheduler, early stopping)
# ---------------------------------------------------------------------------


def train_epoch_finetune(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    verbose: bool = True
) -> Tuple[float, float, float, float, float, float, float]:
    """
    Train fine-tuning model for one epoch.

    Args:
        model: PyTorch model to train
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run on
        verbose: Whether to show progress bar

    Returns:
        Tuple of (loss, accuracy, precision, recall, specificity, mcc, auc)
    """
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    iterator = tqdm(dataloader, desc="Training") if verbose else dataloader

    for batch in iterator:
        protein_input_ids = batch['protein_input_ids'].to(device)
        protein_attention_mask = batch['protein_attention_mask'].to(device)
        molecule_input_ids = batch['molecule_input_ids'].to(device)
        molecule_attention_mask = batch['molecule_attention_mask'].to(device)
        labels = batch['label'].unsqueeze(1).to(device)

        optimizer.zero_grad()

        logits = model(
            protein_input_ids=protein_input_ids,
            protein_attention_mask=protein_attention_mask,
            molecule_input_ids=molecule_input_ids,
            molecule_attention_mask=molecule_attention_mask
        )
        logits = torch.clamp(logits, min=1e-7, max=1-1e-7)

        loss = criterion(logits, labels)
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        preds = (logits > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(logits.cpu().detach().numpy())

    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    specificity = specificity_score(all_labels, all_preds, average='binary', pos_label=1)

    try:
        auc = roc_auc_score(all_labels, all_probs)
        mcc = matthews_corrcoef(all_labels, all_preds)
    except ValueError:
        auc = 0.5
        mcc = 0
        print("Warning: Could not calculate AUC/MCC")

    return avg_loss, accuracy, precision, recall, specificity, mcc, auc


def evaluate_model_finetune(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    verbose: bool = True
) -> Tuple[List, List, float, float, float, float, float, float, float]:
    """
    Evaluate fine-tuning model on a dataset.

    Args:
        model: PyTorch model to evaluate
        dataloader: Evaluation data loader
        criterion: Loss function
        device: Device to run on
        verbose: Whether to show progress bar

    Returns:
        Tuple of (predictions, probabilities, loss, accuracy, precision, recall, specificity, mcc, auc)
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    iterator = tqdm(dataloader, desc="Evaluating") if verbose else dataloader

    with torch.no_grad():
        for batch in iterator:
            protein_input_ids = batch['protein_input_ids'].to(device)
            protein_attention_mask = batch['protein_attention_mask'].to(device)
            molecule_input_ids = batch['molecule_input_ids'].to(device)
            molecule_attention_mask = batch['molecule_attention_mask'].to(device)
            labels = batch['label'].unsqueeze(1).to(device)

            logits = model(
                protein_input_ids=protein_input_ids,
                protein_attention_mask=protein_attention_mask,
                molecule_input_ids=molecule_input_ids,
                molecule_attention_mask=molecule_attention_mask
            )
            logits = torch.clamp(logits, min=1e-7, max=1-1e-7)

            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = (logits > 0.5).float()
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
            all_probs.extend(logits.cpu().numpy().flatten())

    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    specificity = specificity_score(all_labels, all_preds, average='binary', pos_label=1)

    try:
        auc = roc_auc_score(all_labels, all_probs)
        mcc = matthews_corrcoef(all_labels, all_preds)
    except ValueError:
        auc = 0.5
        mcc = 0
        print("Warning: Could not calculate AUC/MCC")

    return all_preds, all_probs, avg_loss, accuracy, precision, recall, specificity, mcc, auc


def train_model_finetune(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    num_epochs: int = 10,
    save_path: str = "best_model_finetune.pth",
    verbose: bool = True,
    early_stopping_patience: int = 5
) -> Tuple[nn.Module, List[dict]]:
    """
    Complete fine-tuning loop with validation, LR scheduling, and early stopping.

    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to run on
        num_epochs: Number of training epochs
        save_path: Path to save best model
        verbose: Whether to print progress
        early_stopping_patience: Number of epochs to wait before early stopping

    Returns:
        Tuple of (best_model, history)
    """
    best_val_auc = 0
    patience_counter = 0
    history = []

    for epoch in range(num_epochs):
        if verbose:
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)

        # Training
        train_loss, train_acc, train_prec, train_rec, train_spec, train_mcc, train_auc = train_epoch_finetune(
            model, train_loader, criterion, optimizer, device, verbose=verbose
        )

        # Validation
        val_preds, val_probs, val_loss, val_acc, val_prec, val_rec, val_spec, val_mcc, val_auc = evaluate_model_finetune(
            model, val_loader, criterion, device, verbose=verbose
        )

        # Step scheduler
        if scheduler is not None:
            scheduler.step(val_mcc)

        # Save history
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_auc': train_auc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_auc': val_auc,
            'val_mcc': val_mcc,
            'learning_rate': optimizer.param_groups[0]['lr']
        })

        if verbose:
            print(f"Train - Loss: {train_loss:.3f}, Acc: {train_acc:.3f}, AUC: {train_auc:.3f}, "
                  f"Sens: {train_rec:.3f}, Spec: {train_spec:.3f}, MCC: {train_mcc:.3f}")
            print(f"Val   - Loss: {val_loss:.3f}, Acc: {val_acc:.3f}, AUC: {val_auc:.3f}, "
                  f"Sens: {val_rec:.3f}, Spec: {val_spec:.3f}, MCC: {val_mcc:.3f}")
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")

        # Save best model and early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
            if verbose:
                print(f"New best model saved! Val AUC: {best_val_auc:.4f}")
        else:
            patience_counter += 1
            if verbose:
                print(f"No improvement. Patience: {patience_counter}/{early_stopping_patience}")

        # Early stopping
        if patience_counter >= early_stopping_patience:
            if verbose:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break

    # Load best model
    model.load_state_dict(torch.load(save_path))

    return model, history
