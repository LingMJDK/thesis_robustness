import os
import time
import random
from datetime import datetime
import numpy as np
import pandas as pd
import pytz

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets

from tqdm import tqdm
from timeit import default_timer as Timer
from utils import save_dataframe_if_not_exists, save_checkpoint, lr_lambda, print_train_time
from train import train_step, train_step_adv_AUGMIX, accuracy_fn
from test import test_step

from datasets import create_train_val_test_ds
from model import ViTClassificationHead
import fire


def main(DATA_SEED: int = 22,
         SEEDS: list = [11, 22, 33], # <------ Determines the number of models to train in this session
         EPOCHS: int = 100,
         BATCH_SIZE: int = 256,
         NUM_WORKERS: int = 10,
         LEARNING_RATE: float = 2e-3,
         WEIGHT_DECAY: float = 5e-2,
         USE_SIMPLE_AUGMIX: bool = False,
         USE_ADVANCED_AUGMIX: bool = True,
         MIXED_PRECISION: bool = True,
         CHECKPOINT_INTERVAL: int = 20,
         MODEL_NAME: str = "ViT_base", # <---------- Reminder, always change model names when changing from ViT_base to MAE
         CHECKPOINT_DIR: str = 'models',
         DF_LOGS_SAVE_DIR: str = 'training_logs',
         DATA_DIR: str = 'data'
         ):
    # _______Set device__________s__
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Needs to be consistent across experiments
    model_config = {"num_classes": 10,
                "in_channels": 3,
                "image_size": 32,
                "patch_size": 4,
                "emb_size": 192,
                "n_layers": 9,
                "n_heads": 12,
                "ff_hidden_mult":4,
                "dropout": 0.1
                }
    
    if device != 'cuda':  # Automically turn mixed precision of when training on cpu
        MIXED_PRECISION = False
    
    # Fix all the RNG seeds for reproducibility
    random.seed(DATA_SEED)
    np.random.seed(DATA_SEED)
    torch.manual_seed(DATA_SEED)

    # Make cuDNN deterministic (this must happen before any cudnn‐based ops)
    cudnn.deterministic = True
    cudnn.benchmark = False
    

    if USE_SIMPLE_AUGMIX:
        MODEL_NAME = f'{MODEL_NAME}_SimAUGMIX'

    if USE_ADVANCED_AUGMIX:
        BATCH_SIZE = BATCH_SIZE // 3
        MODEL_NAME = f'{MODEL_NAME}_AdvAUGMIX'
    
    augmix_config = {
    "severity": 3,
    "width": 3,
    "depth": -1,
    "alpha": 1.0
    }
           

    assert not (USE_SIMPLE_AUGMIX and USE_ADVANCED_AUGMIX), \
        "Make sure only one form of AUGMIX is enabled"
        
    # Params --Checkpoint
    checkpoint_dir_model = os.path.join(CHECKPOINT_DIR, f"checkpoints_{MODEL_NAME}")
    os.makedirs(checkpoint_dir_model, exist_ok=True)

    # Params --Training logs
    tz = pytz.timezone('Europe/Amsterdam')
    time_stamp = datetime.now(tz).strftime('%Y-%m-%d__%H_%M_%S')
    
    df_logs_file_name = f"Train_logs_{MODEL_NAME}_{time_stamp}.csv"
    df_logs_save_dir_model = os.path.join(DF_LOGS_SAVE_DIR, f"train_logs_{MODEL_NAME}")
    os.makedirs(df_logs_save_dir_model, exist_ok=True)

    # 2.) __________Create the Train-, Validation- and Test data_______________

    training_logs = {"model_version": [],
                    "epoch": [],
                    "seed": [],
                    "Learning_rate": [],
                    "train_loss": [],
                    "train_accuracy": [],
                    "val_loss": [],
                    "val_accuracy": [],
                    "GPU": [],
                    "GPU_used_GB": [],
                    "GPU_max_GB": [],
                    "mixed precision": [],
                    "train_time_epoch": [],
                    "train_time_cum": [],
                            }

    train_data, val_data, _ = create_train_val_test_ds(data_seed=DATA_SEED,
                                                            use_simple_augmix=USE_SIMPLE_AUGMIX,
                                                            use_advanced_augmix=USE_ADVANCED_AUGMIX,
                                                            augmix_config=augmix_config,
                                                            root=DATA_DIR,)

    raw_train = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=None)
    CLASS_NAMES = raw_train.classes
    assert model_config["num_classes"] == len(CLASS_NAMES), \
        "num_classes must equal len(train_data.classes)"



    val_dataloader = DataLoader(dataset=val_data,                    #  <-------- Check if val data is correct (100% or 10%)
                                batch_size=BATCH_SIZE,
                                shuffle=False,
                                num_workers=NUM_WORKERS,
                                pin_memory=True
                                )
    


    #_______ Initialize seeds, models, loss_fn, optimzizer (LOOP LEVEL 1) ________________
    for seed in SEEDS:
        print(f"Starting run: seed={seed}, mixed_precision={MIXED_PRECISION}")
        torch.manual_seed(seed)            # seed CPU ops in PyTorch
        torch.cuda.manual_seed(seed)       # seed GPU ops in PyTorch
        np.random.seed(seed)               # seed NumPy (if used anywhere in transforms)
        random.seed(seed)                  # seed Python’s `random` (if used)
        
        
        g = torch.Generator()
        g.manual_seed(seed)
        
        train_dataloader = DataLoader(
            dataset=train_data,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            generator=g
        )


        # 1. Initialize the model
        model = ViTClassificationHead(**model_config).to(device)


        model_version = f"{MODEL_NAME}_S{seed}"

        # 2. Initialize the loss function, optimizer and scheduler
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,           # start LR
            weight_decay=WEIGHT_DECAY  # WD = 0.05
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # 3. Initialize epochs
        epochs = EPOCHS
        train_time_start_on_gpu = Timer()


    # _____________________ Train Validation loop (LOOP LEVEL 2)____________________________
        for epoch in tqdm(range(1, epochs+1)):
            train_time_start_epoch = Timer()

            if not USE_ADVANCED_AUGMIX:
                # Normal train step --> Advanced AUGMIX turned OFF
                train_loss, train_accuracy, current_lr = train_step(model=model,
                                                            train_data=train_dataloader,
                                                            loss_fn=loss_fn,
                                                            accuracy_fn=accuracy_fn,
                                                            optimizer=optimizer,
                                                            epoch=epoch,
                                                            device=device,
                                                            task='Classification',
                                                            scheduler=scheduler,
                                                            use_amp=MIXED_PRECISION
                                                                    )
            else:
                # Altered train step --> Advanced AUGMIX turned ON
                train_loss, train_accuracy, current_lr = train_step_adv_AUGMIX(
                                                            model=model,
                                                            train_data=train_dataloader,
                                                            loss_fn=loss_fn,
                                                            accuracy_fn=accuracy_fn,
                                                            optimizer=optimizer,
                                                            device=device,
                                                            epoch=epoch,
                                                            scheduler=scheduler,
                                                            use_amp=MIXED_PRECISION
                                                        )

            # Validation step
            val_loss, val_accuracy = test_step(model=model,
                                    test_data=val_dataloader,
                                    loss_fn=loss_fn,
                                    accuracy_fn=accuracy_fn,
                                    epoch=epoch,
                                    device=device,
                                    task="Classification")

            # Saves the model every checkpoint interval
            if epoch % CHECKPOINT_INTERVAL == 0:
                checkpoint_state = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'model_config': model_config,   # <— store the dict itself
                    'class_names': CLASS_NAMES,
                    }

                save_checkpoint(checkpoint_state, checkpoint_dir_model, model_version)


            # record GPU data
            if device == "cuda":
                used_mem = torch.cuda.memory_allocated(device) / (1024 ** 3)        # in GB
                max_mem  = torch.cuda.max_memory_allocated(device) / (1024 ** 3)    # in GB
                gpu_name = torch.cuda.get_device_name(device)
            else:
                used_mem = 0.0
                max_mem  = 0.0
                gpu_name = "cpu"

            # Record cumulatative train time
            train_time_end_on_gpu = Timer()
            total_train_time_model = print_train_time(start=train_time_start_on_gpu,
                                                        end=train_time_end_on_gpu,
                                                        device=device)
            # Record train time per epoch
            epoch_end_time = train_time_end_on_gpu - train_time_start_epoch
            print(f"Train time epoch: {epoch_end_time:.4f} seconds")

            # Add data to training logs
            training_logs["model_version"].append(model_version)
            training_logs["epoch"].append(epoch)
            training_logs["seed"].append(seed)
            training_logs["Learning_rate"].append(current_lr)
            training_logs["train_loss"].append(train_loss)
            training_logs["train_accuracy"].append(train_accuracy)
            training_logs["val_loss"].append(val_loss)
            training_logs["val_accuracy"].append(val_accuracy)
            training_logs["GPU"].append(gpu_name)
            training_logs["train_time_cum"].append(total_train_time_model)
            training_logs["train_time_epoch"].append(epoch_end_time)
            training_logs["mixed precision"].append(MIXED_PRECISION)
            training_logs["GPU_used_GB"].append(used_mem)
            training_logs["GPU_max_GB"].append(max_mem)


    #___________________Saving logs_____________________

    # Convert trainig logs to pandas DataFrame
    df_training_logs = pd.DataFrame(training_logs)

    # Save the DataFrame
    save_dataframe_if_not_exists(df_training_logs, df_logs_save_dir_model, df_logs_file_name)



if __name__ == "__main__":
    fire.Fire(main)

    
    