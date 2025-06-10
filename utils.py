import os
import time
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from typing import List, Tuple, Type, Optional, Union

def measure_loading_time(
    num_workers: int,
    batch_size:  int = 128,
    num_batches: int = 100,
):
    """
    Creates a CIFAR-10 DataLoader with the given num_workers and batch_size,
    then times how long it takes to pull `num_batches` batches from it.
    """
    # Simple ToTensor-only pipeline so we only measure IO/worker overhead
    ds = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,      # order doesn’t matter for timing
        num_workers=num_workers,
        pin_memory=True,
    )

    it = iter(loader)      # spawn workers here once
    t0 = time.perf_counter()
    for i in range(num_batches):
        try:
            _ = next(it)
        except StopIteration:
            break
    t1 = time.perf_counter()

    return t1 - t0


def print_train_time(start: float, end: float, device: torch.device = None):
    """Prints difference between start and end time."""
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time



def lr_lambda(epoch):
    if epoch < 10:
        return (epoch+1) / 10       # so at e=0 → 1/10, e=9 → 10/10
    t = (epoch - 10) / (100 - 10)
    return 0.5*(1 + math.cos(math.pi * t))

def lr_lambda_MAE(epoch):
    if epoch < 10:
        return (epoch+1) / 10       # so at e=0 → 1/10, e=9 → 10/10
    t = (epoch - 10) / (200 - 10)
    return 0.5*(1 + math.cos(math.pi * t))

def save_dataframe_if_not_exists(dataframe, save_dir, filename):
  """
  Saves a pandas DataFrame to a specified directory only if the file doesn't
  already exist in that directory.

  Params:
    dataframe (pd.DataFrame): The DataFrame to save.
    save_dir (str): The directory where the DataFrame should be saved.
    filename (str): The name of the file to save the DataFrame as (e.g., 'results.csv').
  """
  full_path = os.path.join(save_dir, filename)

  if not os.path.exists(full_path):
    os.makedirs(save_dir, exist_ok=True) # Create directory if it doesn't exist
    dataframe.to_csv(full_path, index=False)
    print(f"DataFrame saved to {full_path}")
  else:
    print(f"File already exists at {full_path}. DataFrame not saved.")
    

def save_checkpoint(state: dict, checkpoint_dir: str, model_version: str):
    """
    Save training checkpoint without overwriting previous ones.
    Filenames are of the form:
      {model_version}_epoch{epoch}_{YYYYmmdd_HHMMSS}.pth
    Args:
        state: dict with 'epoch', 'model_state_dict', etc.
        checkpoint_dir: where to write
        model_version: prefix for the file
    """
    epoch = state.get('epoch', 'NA')
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.gmtime())
    filename = f"{model_version}_epoch{epoch}_{timestamp}.pth"
    path = os.path.join(checkpoint_dir, filename)
    torch.save(state, path)
    print(f"✔ Checkpoint saved as:\n   {path}")
    
def load_checkpoint(
    checkpoint_path: str,
    model_class: Type[nn.Module],
    device: Union[str, torch.device],
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ) -> Tuple[nn.Module, Optional[int]]:
    """
    Load a checkpoint, rebuild the model from its saved config, restore weights
    (and optionally optimizer/scheduler), and return (model, last_epoch).

    Params:
        checkpoint_path: path to the .pth checkpoint file.
        model_class: the class/constructor to build the model (e.g. ViTClassificationHead).
        device: "cpu" or "cuda" (or torch.device).
        optimizer: if provided, its state will be restored.
        scheduler: if provided, its state will be restored.

    Returns:
        model: the instantiated model with loaded weights, on the target device.
        epoch: the epoch number stored in the checkpoint (or None).
    """
    # 1) load once
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 2) rebuild and move to device
    config = checkpoint['model_config']
    model = model_class(**config).to(device)

    # 3) load model weights
    model.load_state_dict(checkpoint['model_state_dict'])

    # 4) (optional) load optimizer and scheduler
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint.get('epoch', None)
    print(f"Loaded '{os.path.basename(checkpoint_path)}' (epoch {epoch})")

    return model, epoch


def load_models_for_eval(model_paths: List[str],
                         device: str,
                         model_class: Type[nn.Module]
                         ) -> List[Tuple[nn.Module, str]]:
    """
    Load and return models from given checkpoint paths.

    Params:
        model_paths: List of file paths to model checkpoint files.
        device:     Device specifier (e.g., "cpu" or "cuda").
        model_class: Class used to instantiate the model (must match saved config).

    Returns:
        A list of (model, checkpoint_basename) tuples with weights loaded.
    """
    models = []

    for path in model_paths:
        model, _ = load_checkpoint(path, model_class, device)
        models.append((model, os.path.basename(path)))
    
    return models


if __name__ == "__main__":
    # Measure performance for each several worker configs
    for workers in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20]:
        t = measure_loading_time(num_workers=workers)
        print(f"num_workers={workers:>2} → {t:.2f}s for 100 batches")
        