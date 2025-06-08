import os
import pandas as pd
from typing import List, Tuple, Callable

import torch
from torch import nn
from utils import save_dataframe_if_not_exists
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datetime import datetime
import pytz
from datasets import CIFAR10C

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