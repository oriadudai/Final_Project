import os
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import core.config as config
from core.models.reheartnet import ReHeartNet
from core.losses.composite_loss import ClinicalCompositeLoss

WANDB_AVAILABLE = False


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """Run one training epoch. Returns dict with 'loss' and optional components."""
    model.train()
    running_loss   = 0.0
    running_huber  = 0.0
    running_clef   = 0.0
    n_batches      = len(dataloader)
    for ppg, ecg in dataloader:
        ppg, ecg = ppg.to(device), ecg.to(device)
        optimizer.zero_grad()
        predictions = model(ppg)
        loss = criterion(predictions, ecg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()
        # Collect CLEF composite loss components if available
        if hasattr(criterion, "last_huber_loss"):
            running_huber += criterion.last_huber_loss
            running_clef  += criterion.last_clef_loss

    result = {"loss": running_loss / n_batches}
    if running_huber > 0:
        result["huber_loss"] = running_huber / n_batches
        result["clef_loss"]  = running_clef  / n_batches
    return result


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Evaluate model on a validation DataLoader. Returns average batch loss."""
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for ppg, ecg in dataloader:
            ppg, ecg = ppg.to(device), ecg.to(device)
            predictions = model(ppg)
            loss = criterion(predictions, ecg)
            running_loss += loss.item()
    return running_loss / len(dataloader)


def train_fold(
    fold_idx: int,
    fold_subjects: List[str],
    train_loader: DataLoader,
    val_loader: DataLoader,
    clef_encoder: nn.Module,
    device: torch.device,
    epochs: int = config.EPOCHS,
    lr: float = config.LEARNING_RATE,
    lambda_clinical: float = config.LAMBDA_CLINICAL,
    huber_delta: float = config.HUBER_DELTA,
    hidden_size: int = config.HIDDEN_SIZE,
    checkpoint_dir: str = config.CHECKPOINT_DIR,
    use_wandb: bool = False,
    wandb_kwargs: Optional[dict] = None,
    early_stop_patience: Optional[int] = None,  # paper: run all epochs; set int to enable
    lr_patience: int = 10,                       # only used when lr_schedule="plateau"
    model_name: str = "reheartnet",
    loss_type:  str = "clef",
    lr_schedule: str = "linear_decay",   # paper default: ×0.75 every 50 epochs
    resume_checkpoint: Optional[str] = None,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """Train a model for one CV fold.

    A fresh model is created per fold to ensure no information leaks between
    folds. clef_encoder is shared across folds but is frozen throughout.

    Args:
        fold_idx:        Integer index of this fold (0-based).
        fold_subjects:   List of subject IDs in the test set (used for logging).
        train_loader:    DataLoader for the inner training split.
        val_loader:      DataLoader for the inner validation split.
        clef_encoder:    Frozen CLEF encoder (shared, built once outside).
        device:          Compute device.
        epochs:          Maximum training epochs.
        lr:              Initial Adam learning rate.
        lambda_clinical: Weight for CLEF feature loss term.
        huber_delta:     Huber loss delta parameter.
        hidden_size:     BiLSTM hidden units per direction.
        checkpoint_dir:  Directory for saving best-model checkpoints.
        use_wandb:       Whether to log metrics to Weights & Biases.
        early_stop_patience: Epochs without improvement before early stopping.
                         None disables early stopping entirely (runs all epochs).
        lr_patience:     Epochs without val improvement before halving LR.
                         Only used when lr_schedule="plateau".
        model_name:      Architecture to train: "reheartnet", "lstm", or "bilstm"
                         (see core.models.baselines.get_model).
        loss_type:       "clef"  → Huber + CLEF perceptual loss (our method)
                         "huber" → Huber only
                         "mse"   → Plain MSE (original ReHeartNet)
        lr_schedule:     "linear_decay" → multiply by 0.75 every 50 epochs
                                          (paper default; used by all current models)
                         "plateau"      → ReduceLROnPlateau (non-default, adaptive option)

    Returns:
        (trained_model, loss_history) where loss_history is a dict with
        keys "train" and "val" mapping to lists of per-epoch losses.
    """
    from core.models.baselines import get_model  # noqa: PLC0415

    os.makedirs(checkpoint_dir, exist_ok=True)

    model = get_model(model_name, hidden_size=hidden_size).to(device)

    if loss_type == "mse":
        criterion = nn.MSELoss()
    elif loss_type == "huber":
        # Huber only — no CLEF term
        criterion = nn.HuberLoss(delta=huber_delta)
    else:
        # Default: Huber + CLEF perceptual loss
        criterion = ClinicalCompositeLoss(clef_encoder, lambda_clinical, huber_delta).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if lr_schedule == "linear_decay":
        # Multiply LR by 0.75 every 50 epochs — matches Lee et al. original setup
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda epoch: 0.75 ** (epoch // 50)
        )
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=lr_patience
        )

    run_name = f"{model_name}_fold_{fold_idx:02d}"
    if use_wandb and WANDB_AVAILABLE:
        _wkw = wandb_kwargs or {}
        wandb.init(
            project=_wkw.get("project", "ppg2ecg-reheartnet"),
            entity=_wkw.get("entity", None),
            group=_wkw.get("group", None),    # groups all folds of the same model variant
            name=run_name,
            config={
                "model": model_name,
                "loss_type": loss_type,
                "lr_schedule": lr_schedule,
                "fold": fold_idx,
                "test_subjects": fold_subjects,
                "lr": lr,
                "lambda_clinical": lambda_clinical,
                "huber_delta": huber_delta,
                "hidden_size": hidden_size,
                "epochs": epochs,
            },
            reinit=True,
        )

    best_val_loss = float("inf")
    no_improve    = 0
    start_epoch   = 1
    history: Dict[str, List[float]] = {"train": [], "val": []}

    if resume_checkpoint and os.path.exists(resume_checkpoint):
        ckpt = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        best_val_loss = ckpt["val_loss"]
        start_epoch   = ckpt["epoch"] + 1
        if isinstance(scheduler, optim.lr_scheduler.LambdaLR):
            scheduler.last_epoch = ckpt["epoch"]
        print(f"  Resumed from checkpoint: epoch {ckpt['epoch']}, val_loss={best_val_loss:.5f}")

    for epoch in range(start_epoch, epochs + 1):
        train_info = train_one_epoch(model, train_loader, criterion, optimizer, device)
        train_loss = train_info["loss"]
        val_loss   = validate(model, val_loader, criterion, device)

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        print(
            f"[Fold {fold_idx:02d} | Epoch {epoch:03d}/{epochs}] "
            f"train={train_loss:.5f}  val={val_loss:.5f}"
        )

        if use_wandb and WANDB_AVAILABLE:
            log_dict = {
                "fold":       fold_idx,
                "epoch":      epoch,
                "train_loss": train_loss,
                "val_loss":   val_loss,
                "lr":         optimizer.param_groups[0]["lr"],
            }
            import numpy as _np  # noqa: PLC0415  (needed by both blocks below)
            # Loss components (CLEF model only)
            if "huber_loss" in train_info:
                log_dict["loss/huber"]  = train_info["huber_loss"]
                log_dict["loss/clef"]   = train_info["clef_loss"]
            # Every 100 epochs: log a visual ECG reconstruction image
            if (epoch % 100 == 0 or epoch == epochs) and epoch > 0:
                import matplotlib  # noqa: PLC0415
                matplotlib.use("Agg")
                import matplotlib.pyplot as _plt  # noqa: PLC0415
                model.eval()
                ppg_s, ecg_s = next(iter(val_loader))
                with torch.no_grad():
                    pred_s = model(ppg_s[:2].to(device)).squeeze(-1).cpu().numpy()
                true_s = ecg_s[:2].squeeze(-1).numpy()
                n_show = min(2, len(true_s))
                fig, axes = _plt.subplots(n_show, 1, figsize=(10, 3 * n_show))
                if n_show == 1:
                    axes = [axes]
                fs = 125
                for i in range(n_show):
                    t = _np.linspace(0, len(true_s[i]) / fs, len(true_s[i]))
                    axes[i].plot(t, true_s[i],  color="#1f77b4", lw=1.2, alpha=0.8, label="GT")
                    axes[i].plot(t, pred_s[i],  color="#d45f0e", lw=0.9, alpha=0.8,
                                 linestyle="--", label="Pred")
                    denom = float((true_s[i] ** 2).sum())
                    prd_i = _np.sqrt(((true_s[i] - pred_s[i]) ** 2).sum() / denom) * 100 \
                            if denom > 1e-12 else float("nan")
                    axes[i].set_title(f"Val window {i+1}  |  PRD={prd_i:.1f}%",
                                      fontsize=9)
                    axes[i].legend(fontsize=8)
                    axes[i].set_xlabel("Time (s)", fontsize=8)
                    axes[i].tick_params(labelsize=7)
                fig.suptitle(f"Fold {fold_idx:02d} | Epoch {epoch}", fontsize=10)
                _plt.tight_layout()
                log_dict["reconstruction"] = wandb.Image(fig)
                _plt.close(fig)
                model.train()

            # Every 50 epochs: quick PRD + Pearson r on val set.
            # No CLEF encoder, no R-peak detection — fast enough to run every 50 epochs.
            if epoch % 50 == 0 or epoch == epochs:
                import numpy as _np  # noqa: PLC0415
                model.eval()
                t_all, p_all = [], []
                with torch.no_grad():
                    for ppg_v, ecg_v in val_loader:
                        p_all.append(model(ppg_v.to(device)).squeeze(-1).cpu().numpy())
                        t_all.append(ecg_v.squeeze(-1).numpy())
                t_cat = _np.concatenate(t_all).ravel()
                p_cat = _np.concatenate(p_all).ravel()
                denom = float((t_cat ** 2).sum())
                prd_live = float(_np.sqrt(((t_cat - p_cat) ** 2).sum() / denom) * 100) \
                           if denom > 1e-12 else float("nan")
                r_live   = float(_np.corrcoef(t_cat, p_cat)[0, 1]) \
                           if t_cat.std() > 1e-8 and p_cat.std() > 1e-8 else float("nan")
                rmse_live = float(_np.sqrt(_np.mean((t_cat - p_cat) ** 2)))
                log_dict["val_rmse"]      = rmse_live
                log_dict["val_prd"]       = prd_live
                log_dict["val_pearson_r"] = r_live
                model.train()
            wandb.log(log_dict)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve    = 0
            ckpt_path = os.path.join(checkpoint_dir, f"{model_name}_fold_{fold_idx:02d}_best.pt")
            torch.save({
                "fold":                fold_idx,
                "fold_subjects":       fold_subjects,
                "epoch":               epoch,
                "model_state_dict":    model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss":            val_loss,
                "hidden_size":         hidden_size,
            }, ckpt_path)
        else:
            no_improve += 1
            if early_stop_patience is not None and no_improve >= early_stop_patience:
                print(f"  Early stopping at epoch {epoch} (no improvement for {early_stop_patience} epochs)")
                break

    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()

    # Reload best weights before returning
    ckpt_path = os.path.join(checkpoint_dir, f"{model_name}_fold_{fold_idx:02d}_best.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    return model, history


def split_train_val(dataset, val_fraction: float = 0.1, seed: int = 42):
    """Thin wrapper kept here for import convenience — delegates to data_loader."""
    from core.data_loader import split_train_val as _split
    return _split(dataset, val_fraction=val_fraction, seed=seed)
