import os
import pandas as pd
from typing import List, Tuple, Callable
from datetime import datetime

import torch
from torch import nn
from utils import save_dataframe_if_not_exists
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datetime import datetime
import pytz
from datasets import CIFAR10C
from model import unpatchify
import matplotlib.pyplot as plt

def test_step(model, test_data, loss_fn, accuracy_fn, device, epoch, task="unknown"):
    model.eval()
    test_loss, test_acc = 0.0, 0.0
    with torch.inference_mode():
        for X, y in test_data:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y_logits = model(X)
            y_preds = y_logits.argmax(dim=-1)
            if task.lower() == "classification":
                loss = loss_fn(y_logits, y)
                acc = accuracy_fn(y_preds, y)
            else:
                B, T, V = y_logits.shape
                loss = loss_fn(y_logits.view(B*T, V), y.view(B*T))
                acc = accuracy_fn(y_preds, y)
            test_loss += loss.item()
            test_acc  += acc
    n = len(test_data)
    test_loss /= n
    test_acc  /= n
    print(f"Epoch: {epoch} | Test loss: {test_loss:.4f} | Test accuracy {test_acc:.2f}%")
    return test_loss, test_acc

def mae_test_step(
    model: nn.Module,
    epoch: int,
    dataloader: DataLoader,
    device: torch.device,
    vis_interval: int,
    image_size: int,
    mask_ratio: float = 0.75
    ) -> float:
    """
    Single‐epoch evaluation step for a Masked Autoencoder, with optional visualization.

    Params:
      model (nn.Module): MaskedAutoencoderViT instance to evaluate.
      epoch (int): Current epoch number (for deciding when to visualize).
      dataloader (DataLoader): Yields (images, _) per batch.
      device (torch.device): Device to run evaluation on.
      vis_interval (int): Visualize reconstructions every N epochs.
      image_size (int): Height/width of original images for unpatchify.
      mask_ratio (float, optional): Fraction of patches to mask (default=0.75).

    Returns:
      recon_loss (float): Average mean‐squared‐error over all evaluation samples.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.inference_mode():
        for X, _ in dataloader:
            X = X.to(device)
            loss, _, _ = model(X, mask_ratio)
            total_loss += loss.item() * X.size(0)
            total_samples += X.size(0)

    recon_loss = total_loss / total_samples
    print(f"Reconstruction Loss (Average MSE): {recon_loss:.4f}")

    # Visualize reconstructions at specified interval
    if epoch % vis_interval == 0:
        X_vis, _ = next(iter(dataloader))
        X_vis_cpu = X_vis.cpu()

        with torch.inference_mode():
            _, pred_patches, _ = model(X_vis.to(device), mask_ratio)
        recon = unpatchify(
            pred_patches.cpu(),
            model.patch_size,
            model.in_channels,
            image_size
        )
        plot_reconstructions(X_vis_cpu, recon, n=8)

    return recon_loss

def evaluate_cifar10c(
    model: torch.nn.Module,
    data_root: str,
    corruptions: list,
    transform,
    batch_size: int,
    num_workers: int,
    device,
    severities: list = [None],  # pass [1,2,3,4,5] for per-level
):
    """
    Returns results[c][s] = accuracy%, and
            mean_acc[s]  = mean over corruptions at severity s.
    """
    model.eval()
    results = {c: {} for c in corruptions}
    mean_acc = {}

    with torch.no_grad():
        for s in severities:
            accs = []
            for c in corruptions:
                ds = CIFAR10C(data_root, c, transform=transform, severity=s)
                loader = DataLoader(
                    ds,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                )
                correct = total = 0
                for imgs, labels in loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    preds = model(imgs).argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total   += labels.size(0)
                acc = correct / total * 100
                results[c][s] = acc
                accs.append(acc)
            mean_acc[s] = sum(accs) / len(accs)

    return results, mean_acc


def compute_ece(model, data_loader, device, n_bins=10):
    model.eval()
    confidences = []
    predictions = []
    labels      = []
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            probs  = F.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
            confidences.append(conf)
            predictions.append(pred)
            labels.append(y)
    confidences = torch.cat(confidences)
    predictions = torch.cat(predictions)
    labels      = torch.cat(labels)
    N = len(labels)

    # Bin edges: [0, 1/M, 2/M, …,1]
    bins = torch.linspace(0, 1, steps=n_bins+1, device=device)
    ece  = torch.zeros(1, device=device)

    for i in range(n_bins):
        # select samples in bin i
        in_bin = (confidences > bins[i]) & (confidences <= bins[i+1])
        prop_in_bin = in_bin.float().mean()  # n_m / N
        if prop_in_bin.item() > 0:
            acc_in_bin  = (predictions[in_bin] == labels[in_bin]).float().mean()
            conf_in_bin = confidences[in_bin].mean()
            ece += torch.abs(conf_in_bin - acc_in_bin) * prop_in_bin

    return ece.item() * 100  # in percentage points



def eval_models_clean_test(models: List[Tuple[nn.Module, str]],
                           test_dataloader: DataLoader,
                           loss_fn: nn.Module,
                           accuracy_fn: Callable,
                           device: str,
                           task: str = "Classification",
                           save_dir=None
                           ) -> pd.DataFrame:
    """
    Evaluate a list of models on a clean test set and return results as a DataFrame.

    Params:
        models:           List of (model, model_name) tuples to evaluate.
        test_dataloader:  DataLoader for the clean test dataset.
        loss_fn:          Loss function to use (e.g., nn.CrossEntropyLoss()).
        accuracy_fn:      Function that computes accuracy given logits and targets.
        device:           Device specifier, e.g. "cpu" or "cuda".
        task:             Name of the task (for logging/print), default "Classification".

    Returns:
        A pandas DataFrame with columns ["model_name", "val_loss", "val_accuracy"].
    """

    results = []
    for model, model_name in models:
        val_loss, val_acc = test_step(
            model=model,
            test_data=test_dataloader,
            loss_fn=loss_fn,
            accuracy_fn=accuracy_fn,
            epoch=1,
            device=device,
            task=task
        )
        results.append({
            "model_name":   model_name,
            "val_loss":     val_loss,
            "val_accuracy": val_acc
        })
    
    df_clean = pd.DataFrame(results)

    if not save_dir:
      print("WARNING, no save directory specified, the resulting dataframe will not be saved")
    else:
      tz = pytz.timezone('Europe/Amsterdam')
      time_stamp = datetime.now(tz).strftime('%Y-%m-%d__%H_%M_%S')
      model_name = models[0][1]

      if "_S" in model_name:
          # Use only the base model name
          base = model_name.split("_S", 1)[0]
      else:
          # cut off extension
          base = os.path.splitext(model_name)[0]

      file_name = f"TESTcifar10_clean__results_{base}_{time_stamp}.csv"
      
      save_dataframe_if_not_exists(df_clean, save_dir, file_name)

    return df_clean


def plot_reconstructions(imgs,
                         recon_imgs,
                         n=8,
                         mean: tuple = (0.485, 0.456, 0.406),
                         std: tuple  = (0.229, 0.224, 0.225),
                         root: str = "data/plots"
                         ):
    """
    Plots the first n original images and their reconstructions side by side,
    with denormalization for CIFAR-10. Saves the plot to a unique file in `root`.

    Args:
      imgs:       Tensor of shape (B, C, H, W), normalized images.
      recon_imgs: Tensor of shape (B, C, H, W), normalized reconstructions.
      n:          Number of examples to display.
      root:       Directory to save plots (default: 'data/plots').
    """

    os.makedirs(root, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(root, f"reconstructions_{timestamp}.png")

    imgs = imgs.cpu()
    recon_imgs = recon_imgs.cpu()

    # Get shapes
    B, C, H, W = imgs.shape

    mean = torch.tensor(mean, dtype=imgs.dtype, device=imgs.device).reshape(1, C, 1, 1)
    std  = torch.tensor(std, dtype=imgs.dtype, device=imgs.device).reshape(1, C, 1, 1)

    imgs    = imgs * std + mean
    recon_imgs = recon_imgs * std + mean

    imgs = torch.clamp(imgs, 0, 1)
    recon_imgs = torch.clamp(recon_imgs, 0, 1)

    plt.figure(figsize=(n*2, 4))
    for i in range(min(n, B)): # Ensure we don't exceed batch size
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(imgs[i].permute(1, 2, 0).numpy())
        ax.set_title("Original")
        ax.axis("off")

        # Reconstructed image
        ax = plt.subplot(2, n, n + i + 1)
        plt.imshow(recon_imgs[i].permute(1, 2, 0).numpy())
        ax.set_title("Reconstructed")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved reconstructions to {save_path}")