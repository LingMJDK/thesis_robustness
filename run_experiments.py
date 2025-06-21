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
from utils import save_dataframe_if_not_exists, save_checkpoint, lr_lambda, print_train_time, lr_lambda_MAE
from train import train_step, train_step_adv_AUGMIX, mae_train_step, accuracy_fn
from test import test_step, mae_test_step
from typing import List

from datasets import create_train_val_test_ds, create_pretrain_loaders
from model import ViTClassificationHead, MaskedAutoencoderViT, build_finetune_vit_from_mae
import fire

def run_experiment(
    TRAIN_MODE: str = 'MAE_finetune',
    PRETRAIN_CKPT_FILENAME: List[str] = ["MAE_ViT_PreTrain_S11_epoch200_20250618_184953.pth", "MAE_ViT_PreTrain_S22_epoch200_20250618_222624.pth", "MAE_ViT_PreTrain_S33_epoch200_20250619_015442.pth"],
    UNFREEZE: int = 200,
    DROPOUT_MAE: float = 0.0,  # decreased from 0.1
    SEEDS: list = [11], # [11, 22, 33],
    EPOCHS: int = 10,
    EPOCHS_MAE: int = 30,
    BATCH_SIZE: int = 256,
    BATCH_SIZE_PRE_TRAIN: int = 256,
    PREFETCH_FACTOR: int = 4,
    PERSISTENT_WORKERS: bool = False,
    NUM_WORKERS: int = 12,
    LEARNING_RATE: float = 2e-3,
    LEARNING_RATE_MAE: float = 1e-4,
    WEIGHT_DECAY: float = 5e-2,
    WEIGHT_DECAY_MAE: float = 5e-2,
    USE_SIMPLE_AUGMIX: bool = False,
    USE_ADVANCED_AUGMIX: bool = False,
    MIXED_PRECISION: bool = True,
    CHECKPOINT_INTERVAL: int = 20,
    MODEL_NAME: str = "ViT_base",
    CHECKPOINT_DIR: str = 'models',
    DF_LOGS_SAVE_DIR: str = 'training_logs',
    DATA_DIR: str = 'data',
    MASK_RATIO: float = 0.75,
    DECODER_LAYERS: int = 4,
    VIS_INTERVAL: int = 2,
    PATCH_SIZE_PRETRAIN: int = 4,
    IMAGE_SIZE_PRETRAIN: int = 32,
    PRETRAIN_MEAN_C_STAT = (0.4914, 0.4822, 0.4465),
    PRETRAIN_STD_C_STAT = (0.2470, 0.2435, 0.2616),
    ):

    train_modes = ['ViT_base',
                   'MAE_pretrain',
                   'MAE_finetune']
    
    augmix_config = {
    "severity": 3,
    "width": 3,
    "depth": -1,
    "alpha": 1.0
    }




    assert TRAIN_MODE != None and TRAIN_MODE in train_modes, \
    f'please select one of the following train_modes: {train_modes}'

    if TRAIN_MODE == 'MAE_pretrain':
      mean = PRETRAIN_MEAN_C_STAT
      std  = PRETRAIN_STD_C_STAT
    else: # CIFAR 10 channel statistics
      mean = (0.4914, 0.4822, 0.4465)
      std  = (0.2470, 0.2435, 0.2616)

    peak_lr_mae = LEARNING_RATE_MAE * BATCH_SIZE_PRE_TRAIN / 256
    # peak_lr_mae = LEARNING_RATE_MAE

    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != 'cuda':
        MIXED_PRECISION = False

    # Base configuration dict
    model_config = {
        "num_classes":    10,
        "in_channels":    3,
        "image_size":     32,
        "patch_size":     4,  # <------ finetune on patch_size=4 for 32x32 = grid of 64 tokens
        "emb_size":       192,
        "n_layers":       9,
        "n_heads":        12,
        "ff_hidden_mult": 4,
        "dropout":        0.1
    }

    mae_config = {
        "in_channels":    3,
        "image_size":     IMAGE_SIZE_PRETRAIN,  # <------ Updated to 96 for pre-training
        "patch_size":     PATCH_SIZE_PRETRAIN,  # <------ Pre-train on patch_size=12 for 96x96 = grid of 64 tokens
        "emb_size":       192,
        "encoder_layers": 9,
        "n_heads":        12,
        "ff_hidden_mult": 4,
        "decoder_layers": DECODER_LAYERS,
        "dropout":        DROPOUT_MAE
    }


    if TRAIN_MODE == 'MAE_pretrain':
      MODEL_NAME = f'MAE_ViT_PreTrain'
    elif TRAIN_MODE == 'MAE_finetune':
      MODEL_NAME = f'MAE_ViT_FineTune'
    # AugMix adjustments for fine-tuning
    if USE_SIMPLE_AUGMIX:
        MODEL_NAME = f'{MODEL_NAME}_SimAUGMIX'
    if USE_ADVANCED_AUGMIX:
        BATCH_SIZE //= 3
        MODEL_NAME = f'{MODEL_NAME}_AdvAUGMIX'


    # Prepare log structure (include phase column)
    training_logs = {
        "phase": [],
        "model_version": [],
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
        "train_time_cum": []
    }

    # Timestamp for saving logs
    tz = pytz.timezone('Europe/Amsterdam')
    time_stamp = datetime.now(tz).strftime('%Y-%m-%d__%H_%M_%S')

    # Create directories
    checkpoint_dir_model = os.path.join(CHECKPOINT_DIR, f"checkpoints_{MODEL_NAME}")
    os.makedirs(checkpoint_dir_model, exist_ok=True)
    df_logs_save_dir_model = os.path.join(DF_LOGS_SAVE_DIR, f"train_logs_{MODEL_NAME}")
    os.makedirs(df_logs_save_dir_model, exist_ok=True)
    df_logs_file_name = f"Train_logs_{MODEL_NAME}_{time_stamp}.csv"

    # Load raw data once (for class names)
    raw_train = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=None)
    CLASS_NAMES = raw_train.classes

    for seed in SEEDS:
        # Prepare data splits and loaders
        train_data, val_data, _ = create_train_val_test_ds(
            data_seed=seed,
            use_simple_augmix=USE_SIMPLE_AUGMIX,
            use_advanced_augmix=USE_ADVANCED_AUGMIX,
            augmix_config=augmix_config,
            root=DATA_DIR,
            mean=mean,
            std=std
        )

        if TRAIN_MODE in ('ViT_base', 'MAE_finetune'):
          val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False,
                                      num_workers=NUM_WORKERS, pin_memory=True)
          g = torch.Generator().manual_seed(seed)
          train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                                        num_workers=NUM_WORKERS, pin_memory=True, generator=g)
          
        else:
          pre_train_loader, pre_val_loader = create_pretrain_loaders(root=DATA_DIR,
                                                                    image_size=IMAGE_SIZE_PRETRAIN,
                                                                    batch_size=BATCH_SIZE_PRE_TRAIN,
                                                                    num_workers=NUM_WORKERS,
                                                                    split_seed=seed,
                                                                    prefetch_factor=PREFETCH_FACTOR,
                                                                    persistent_workers=PERSISTENT_WORKERS
                                                                    )
          batch = next(iter(pre_train_loader))
          print("Got one batch:", batch[0].shape)

        # Seed everything
        print(f"Starting run: seed={seed}, mixed_precision={MIXED_PRECISION}... ")
        if TRAIN_MODE == "MAE_pretrain":
          print(f"Training for {EPOCHS_MAE} epochs in {TRAIN_MODE} mode... ")
          print(f'printing image reconstruction every {VIS_INTERVAL} epochs... ')
        elif USE_ADVANCED_AUGMIX:
          print(f"Training for {EPOCHS} epochs in {TRAIN_MODE} mode with AUGMIX enabled... ")
        else:
          print(f"Training for {EPOCHS} epochs in {TRAIN_MODE} mode with AUGMIX disabled... ")
        print(f'Saving model every {CHECKPOINT_INTERVAL} epochs...\n\n ')


        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False


        model_version = f"{MODEL_NAME}_S{seed}"

        if TRAIN_MODE == 'MAE_pretrain':
          # -------- MAE Pre-training --------
          mae = MaskedAutoencoderViT(**mae_config).to(device)


          # # Collect encoder parameters explicitly
          # encoder_params = list(mae.patch_embed.parameters()) + \
          #                  [mae.pos_emb_enc] + \
          #                  list(mae.encoder_blocks.parameters()) + \
          #                  list(mae.enc_norm.parameters())

          # mae_optimizer = torch.optim.AdamW(mae.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

          mae_optimizer = torch.optim.AdamW(mae.parameters(), lr=peak_lr_mae, weight_decay=WEIGHT_DECAY_MAE)
          mae_scheduler = torch.optim.lr_scheduler.LambdaLR(mae_optimizer, lr_lambda_MAE)


          pre_train_start = Timer()
          for epoch in range(1, EPOCHS_MAE + 1):
              epoch_start = Timer()
              phase = 'pre-training'
              # Train MSE and current learning rate
              train_mse, current_lr = mae_train_step(model=mae,
                                      epoch=epoch,
                                      scheduler=mae_scheduler,
                                      optimizer=mae_optimizer,
                                      device=device,
                                      mask_ratio=MASK_RATIO,
                                      train_dataloader=pre_train_loader,
                                      use_amp=MIXED_PRECISION)

              # Validation MSE
              val_mse = mae_test_step(model=mae,
                        epoch=epoch,
                        device=device,
                        dataloader=pre_val_loader,
                        vis_interval=VIS_INTERVAL,
                        image_size=mae_config["image_size"],
                        mask_ratio=MASK_RATIO)


              # Timing & GPU stats
              epoch_end = Timer()
              cum_time = print_train_time(start=pre_train_start, end=epoch_end, device=device)
              single_time = epoch_end - epoch_start
              if device == "cuda":
                  used_mem = torch.cuda.memory_allocated() / 1e9
                  max_mem  = torch.cuda.max_memory_allocated() / 1e9
                  gpu_name = torch.cuda.get_device_name(device)
              else:
                  used_mem = max_mem = 0.0; gpu_name = "cpu"

              # Log MAE metrics
              training_logs["phase"].append(phase)
              training_logs["model_version"].append(model_version)
              training_logs["epoch"].append(epoch)
              training_logs["seed"].append(seed)
              training_logs["Learning_rate"].append(current_lr)
              training_logs["train_loss"].append(train_mse)
              training_logs["train_accuracy"].append(np.nan)
              training_logs["val_loss"].append(val_mse)
              training_logs["val_accuracy"].append(np.nan)
              training_logs["mixed precision"].append(MIXED_PRECISION)
              training_logs["GPU"].append(gpu_name)
              training_logs["GPU_used_GB"].append(used_mem)
              training_logs["GPU_max_GB"].append(max_mem)
              training_logs["train_time_epoch"].append(single_time)
              training_logs["train_time_cum"].append(cum_time)

                        # Checkpoint
              if epoch % CHECKPOINT_INTERVAL == 0:
                  checkpoint_state = {
                      'epoch': epoch,
                      'model_state_dict': mae.state_dict(),
                      'optimizer_state_dict': mae_optimizer.state_dict(),
                      'scheduler_state_dict': mae_scheduler.state_dict(),
                      'model_config': mae_config,
                      'class_names': CLASS_NAMES,
                  }
                  save_checkpoint(checkpoint_state, checkpoint_dir_model, model_version)



        # -------- ViT Fine-tuning --------      # <------------ UNDER CONSTRUCTION (NEED TO LOAD IN PRETRAINED MODELS, 1 for each seed)
        # Build classifier from MAE encoder
        elif TRAIN_MODE == "MAE_finetune":
          # Find the right checkpoint file for this seed
          ckpt_dir = os.path.join(CHECKPOINT_DIR, f"checkpoints_MAE_ViT_PreTrain")
          ckpt_path = os.path.join(ckpt_dir, PRETRAIN_CKPT_FILENAME[int((str(seed))[0]) - 1])
          assert os.path.isfile(ckpt_path), f"No such file {ckpt_path}"

          # Load checkpoint into MAE and build ViT
          mae = MaskedAutoencoderViT(**mae_config).to(device)
          ckpt = torch.load(ckpt_path, map_location=device)
          mae.load_state_dict(ckpt["model_state_dict"])
          
          vit = build_finetune_vit_from_mae(mae, 
                                            num_classes=model_config["num_classes"], 
                                            dropout=model_config["dropout"],
                                            patch_size=model_config['patch_size']).to(device)
          phase = 'fine-tuning'
          # Optionally freeze encoder first 5 epochs
          for param in vit.encoder.parameters():
              param.requires_grad = False



        # _____Build model without pre-training if MAE is False__________
        elif TRAIN_MODE == 'ViT_base':
          vit = ViTClassificationHead(**model_config).to(device)
          phase = "no pre-training"


        if TRAIN_MODE != 'MAE_pretrain':
          classifier_optimizer = torch.optim.AdamW(vit.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
          classifier_scheduler = torch.optim.lr_scheduler.LambdaLR(classifier_optimizer, lr_lambda)
          loss_fn = nn.CrossEntropyLoss()

          fine_tune_start = Timer()
          for epoch in range(1, EPOCHS + 1):
              epoch_start = Timer()

              # Supervised train (with AugMix if configured)
              if USE_ADVANCED_AUGMIX:
                  train_loss, train_acc, current_lr = train_step_adv_AUGMIX(model=vit,
                                                                  train_data=train_dataloader,
                                                                  loss_fn=loss_fn,
                                                                  accuracy_fn=accuracy_fn,
                                                                  optimizer=classifier_optimizer,
                                                                  device=device,
                                                                  epoch=epoch,
                                                                  scheduler=classifier_scheduler,
                                                                  use_amp=MIXED_PRECISION
                  )
              else:
                  train_loss, train_acc, current_lr = train_step(model=vit,
                                                        train_data=train_dataloader,
                                                        loss_fn=loss_fn,
                                                        accuracy_fn=accuracy_fn,
                                                        optimizer=classifier_optimizer,
                                                        device=device,
                                                        epoch=epoch,
                                                        task='Classification',
                                                        scheduler=classifier_scheduler,
                                                        use_amp=MIXED_PRECISION
                  )

              # Unfreeze after 5 epochs (in case of MAE pre-training)
              if epoch == UNFREEZE and TRAIN_MODE == 'MAE_finetune':
                  for param in vit.encoder.parameters():
                      param.requires_grad = True
                  classifier_optimizer = torch.optim.AdamW(vit.parameters(), lr=LEARNING_RATE_MAE, weight_decay=WEIGHT_DECAY)
                  # Set initial_lr for each param group
                  for param_group in classifier_optimizer.param_groups:
                      param_group['initial_lr'] = LEARNING_RATE_MAE
                  classifier_scheduler = torch.optim.lr_scheduler.LambdaLR(classifier_optimizer, lr_lambda, last_epoch=epoch - 1)
                  print(f"Unfroze encoder and set LR to {LEARNING_RATE_MAE} at epoch {epoch}")
                      

              # Validation
              val_loss, val_acc = test_step(model=vit,
                                            test_data=val_dataloader,
                                            loss_fn=loss_fn,
                                            accuracy_fn=accuracy_fn,
                                            device=device,
                                            epoch=epoch,
                                            task="Classification"
              )

              # Checkpoint
              if epoch % CHECKPOINT_INTERVAL == 0:
                  checkpoint_state = {
                      'epoch': epoch,
                      'model_state_dict': vit.state_dict(),
                      'optimizer_state_dict': classifier_optimizer.state_dict(),
                      'scheduler_state_dict': classifier_scheduler.state_dict(),
                      'model_config': model_config,
                      'class_names': CLASS_NAMES,
                  }
                  save_checkpoint(checkpoint_state, checkpoint_dir_model, model_version)

              # Timing & GPU stats
              epoch_end = Timer()
              cum_time = print_train_time(start=fine_tune_start, end=epoch_end, device=device)
              single_time = epoch_end - epoch_start
              if device == "cuda":
                  used_mem = torch.cuda.memory_allocated() / 1e9
                  max_mem  = torch.cuda.max_memory_allocated() / 1e9
                  gpu_name = torch.cuda.get_device_name(device)
              else:
                  used_mem = max_mem = 0.0

              # Log fine-tune metrics
              training_logs["phase"].append(phase)
              training_logs["model_version"].append(model_version)
              training_logs["epoch"].append(epoch)
              training_logs["seed"].append(seed)
              training_logs["Learning_rate"].append(current_lr)
              training_logs["train_loss"].append(train_loss)
              training_logs["train_accuracy"].append(train_acc)
              training_logs["val_loss"].append(val_loss)
              training_logs["val_accuracy"].append(val_acc)
              training_logs["mixed precision"].append(MIXED_PRECISION)
              training_logs["GPU"].append(gpu_name)
              training_logs["GPU_used_GB"].append(used_mem)
              training_logs["GPU_max_GB"].append(max_mem)
              training_logs["train_time_epoch"].append(single_time)
              training_logs["train_time_cum"].append(cum_time)

    # After all seeds, save logs
    df = pd.DataFrame(training_logs)
    save_dataframe_if_not_exists(df, df_logs_save_dir_model, df_logs_file_name)
if __name__ == "__main__":
  
  fire.Fire(run_experiment)