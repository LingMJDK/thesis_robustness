import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from typing import Callable


def train_step(
    model: torch.nn.Module,
    train_data: torch.utils.data.DataLoader,
    loss_fn,
    accuracy_fn,
    optimizer,
    device: torch.device,
    epoch: int,
    task: str = "unknown",
    scheduler=None,
    use_amp: bool = False,   # <----- new flag for mixed precision
):
    """
    Standard training step with optional AMP.

    Args:
      model       : nn.Module already on `device`
      train_data  : DataLoader yielding (X, y) pairs
      loss_fn     : e.g. nn.CrossEntropyLoss()
      accuracy_fn : function(y_preds, y_true) → percentage
      optimizer   : torch.optim.Optimizer
      device      : torch.device("cuda") or ("cpu")
      epoch       : current epoch (for logging)
      task        : "classification" or "autoregressive" (decides loss/acc logic)
      scheduler   : optional LR scheduler; stepped once per epoch
      use_amp     : if True, wrap forward/backward in torch.cuda.amp.autocast

    Returns:
      (train_loss, train_accuracy) averaged over all batches.
    """
    model.train()
    running_loss = 0.0
    running_acc  = 0.0
    n_batches    = 0

    # Create a GradScaler if AMP is enabled
    scaler = GradScaler() if use_amp else None

    for X, y in train_data:
        # 1) Move input and target to device
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # 2) Forward + loss under autocast if AMP is on
        if use_amp:
            with autocast():
                y_logits = model(X)
                y_preds  = y_logits.argmax(dim=-1)

                if task.lower() == "classification":
                    loss = loss_fn(y_logits, y)
                    acc  = accuracy_fn(y_preds, y)
                else:
                    B, T, V = y_logits.shape
                    loss = loss_fn(y_logits.view(B*T, V), y.view(B*T))
                    acc  = accuracy_fn(y_preds, y)

            # 3) Backprop + optimizer via scaler
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            # Pure FP32 path
            optimizer.zero_grad()
            y_logits = model(X)
            y_preds  = y_logits.argmax(dim=-1)

            if task.lower() == "classification":
                loss = loss_fn(y_logits, y)
                acc  = accuracy_fn(y_preds, y)
            else:
                B, T, V = y_logits.shape
                loss = loss_fn(y_logits.view(B*T, V), y.view(B*T))
                acc  = accuracy_fn(y_preds, y)

            loss.backward()
            optimizer.step()

        # 4) Accumulate loss & accuracy
        running_loss += loss.item()
        running_acc  += acc
        n_batches    += 1

    # Current LR for logging

    current_lr = optimizer.param_groups[0]["lr"]
    # 5) Step scheduler once per epoch
    if scheduler is not None:
        scheduler.step()

    # 6) Compute epoch averages
    train_loss = running_loss / n_batches
    train_acc  = running_acc  / n_batches



    print(
        f"Epoch: {epoch} | Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: {train_acc:.2f}% | lr: {current_lr:.6f}"
    )

    return train_loss, train_acc, current_lr


def train_step_adv_AUGMIX(
    model: torch.nn.Module,
    train_data: torch.utils.data.DataLoader,
    loss_fn,
    accuracy_fn,
    optimizer,
    device: torch.device,
    epoch: int,
    scheduler=None,
    use_amp: bool = False,          # <— new flag
):
    """
    Three‐view AugMix training step with optional AMP.

    Expects train_data to yield tuples of (img_clean, img_aug1, img_aug2, labels),
    where each img_* is already a normalized FloatTensor [B,3,32,32].

    Args:
      model       : nn.Module on `device`
      train_data  : DataLoader → (img_clean, img_aug1, img_aug2, labels)
      loss_fn     : e.g. nn.CrossEntropyLoss()
      accuracy_fn : fn(y_preds, y_true) → percentage
      optimizer   : torch.optim.Optimizer
      device      : torch.device("cuda") or ("cpu")
      epoch       : current epoch (for printing/log)
      scheduler   : optional LR scheduler; stepped once per epoch
      use_amp     : if True, run forward/backward under torch.cuda.amp

    Returns:
      (train_loss, train_accuracy) averaged over all batches.
    """
    model.train()
    epoch_loss = 0.0
    epoch_acc  = 0.0
    n_batches  = 0

    # If using AMP, create a GradScaler. If not, we ignore it.
    scaler = GradScaler() if use_amp else None

    for img_clean, img_aug1, img_aug2, labels in train_data:
        B = img_clean.size(0)

        # 1. Move all tensors to device
        img_clean = img_clean.to(device, non_blocking=True)
        img_aug1  = img_aug1.to(device,  non_blocking=True)
        img_aug2  = img_aug2.to(device,  non_blocking=True)
        labels    = labels.to(device,    non_blocking=True)

        # 2) Forward + loss under autocast if AMP is enabled
        if use_amp:
            with autocast():
                logits_clean = model(img_clean)  # [B, num_classes]
                logits_aug1  = model(img_aug1)
                logits_aug2  = model(img_aug2)

                # CE on clean & aug1
                ce_clean = loss_fn(logits_clean, labels)
                ce_aug1  = loss_fn(logits_aug1,  labels)
                ce_loss  = 0.5 * ce_clean + 0.5 * ce_aug1

                # JS-consistency across three outputs
                p_clean = torch.softmax(logits_clean, dim=1)
                p_aug1  = torch.softmax(logits_aug1,  dim=1)
                p_aug2  = torch.softmax(logits_aug2,  dim=1)
                p_mean  = (p_clean + p_aug1 + p_aug2) / 3.0

                def kl_div(a, b):
                    return torch.sum(a * (torch.log(a + 1e-8) - torch.log(b + 1e-8)), dim=1)

                js_per_example = (
                    kl_div(p_clean, p_mean) +
                    kl_div(p_aug1,  p_mean) +
                    kl_div(p_aug2,  p_mean)
                ) / 3.0
                js_loss = js_per_example.mean()

                LAMBDA = 12.0
                total_loss = ce_loss + LAMBDA * js_loss

            # 3) Backprop & optimizer.step via GradScaler
            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            # Pure FP32 path
            logits_clean = model(img_clean)
            logits_aug1  = model(img_aug1)
            logits_aug2  = model(img_aug2)

            ce_clean = loss_fn(logits_clean, labels)
            ce_aug1  = loss_fn(logits_aug1,  labels)
            ce_loss  = 0.5 * ce_clean + 0.5 * ce_aug1

            p_clean = torch.softmax(logits_clean, dim=1)
            p_aug1  = torch.softmax(logits_aug1,  dim=1)
            p_aug2  = torch.softmax(logits_aug2,  dim=1)
            p_mean  = (p_clean + p_aug1 + p_aug2) / 3.0

            def kl_div(a, b):
                return torch.sum(a * (torch.log(a + 1e-8) - torch.log(b + 1e-8)), dim=1)

            js_per_example = (
                kl_div(p_clean, p_mean) +
                kl_div(p_aug1,  p_mean) +
                kl_div(p_aug2,  p_mean)
            ) / 3.0
            js_loss = js_per_example.mean()

            LAMBDA = 12.0
            total_loss = ce_loss + LAMBDA * js_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        # 4.) Track loss & “clean” accuracy
        epoch_loss += total_loss.item()
        preds_clean = logits_clean.argmax(dim=1)
        correct_clean = (preds_clean == labels).sum().item()
        epoch_acc += (correct_clean / B) * 100.0
        n_batches += 1

    # 5.) Grab current LR for logging
    learning_rate = optimizer.param_groups[0]["lr"]

    # 6) Step scheduler once per epoch (outside batch loop)
    if scheduler is not None:
        scheduler.step()

    # 7) Compute epoch averages
    train_loss = epoch_loss / n_batches
    train_accuracy = epoch_acc / n_batches



    print(
        f"Epoch: {epoch} | Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.2f}% | lr: {learning_rate:.6f}"
    )

    return train_loss, train_accuracy, learning_rate


def accuracy_fn(y_pred, y_true):
    """Calculates accuracy between truth labels and predictions.

    Params:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


def accuracy_fnV2(y_pred, y_true):
    """Returns accuracy as percentage between predictions and labels."""
    correct = (y_pred == y_true).sum().item()
    total = y_true.numel()
    return (correct / total) * 100