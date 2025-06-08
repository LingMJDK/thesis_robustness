import os
import urllib.request
import tarfile
import torch
import numpy as np
from augmix import AugMix
from PIL import Image
from torch.utils.data import Dataset, random_split
import torchvision.transforms as T
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader

class CIFAR10C(Dataset):
    def __init__(
        self,
        data_root: str,
        corruption: str,
        transform=None,
        severity: int = None,  # 1..5 or None for all
    ):
        """
        data_root: folder containing CIFAR-10-C/*.npy
        corruption: e.g. "gaussian_noise"
        severity: Which severity level (1–5), or None to include all 50k images
        """
        base = os.path.join(data_root, corruption + ".npy")
        self.imgs = np.load(base, mmap_mode="r")           # shape (50000,32,32,3)
        self.labels = np.load(os.path.join(data_root, "labels.npy"), mmap_mode="r")
        if severity is not None:
            assert 1 <= severity <= 5
            start = (severity - 1) * 10000
            end   = severity * 10000
            self.imgs   = self.imgs[start:end]
            self.labels = self.labels[start:end]
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img   = Image.fromarray(self.imgs[idx])
        label = int(self.labels[idx])
        if self.transform:
            img = self.transform(img)
        return img, label

class WrappedSubset(Dataset):
    def __init__(self, base_subset, transform):
        self.base_subset = base_subset
        self.transform   = transform

    def __len__(self):
        return len(self.base_subset)

    def __getitem__(self, idx):
        img, label = self.base_subset[idx]       # returns PIL Image, int label
        if self.transform:
            img = self.transform(img)
        return img, label

    def __getattr__(self, name):
      """
      Any attribute not found on WrappedSubset
      will be looked up on the underlying CIFAR10 object, e.g. .classes, .targets, etc.
      """
      # base_subset.dataset is the original CIFAR10 instance
      return getattr(self.base_subset.dataset, name)

class ThreeViewCIFAR(Dataset):
    """
    Wraps a CIFAR10 subset (a torch.utils.data.Subset of raw_train),
    and returns a PIL image + label. We will do AugMix twice per sample in the loop.
    """
    def __init__(self, base_subset):
        """
        base_subset: a torch.utils.data.Subset of a torchvision.datasets.CIFAR10
                     that returns (PIL, label).
        """
        self.base = base_subset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        # Returns a PIL image (no transform) and the label
        pil_img, label = self.base[idx]
        return pil_img, label

def threeview_collate_fn(batch):
    pil_images, labels = zip(*batch)
    return list(pil_images), torch.tensor(labels, dtype=torch.long)


class ThreeViewCIFAR_Tensor(Dataset):
    def __init__(self, base_subset, augmix_config):
        self.base   = base_subset
        self.augmix = AugMix(**augmix_config)  # create once per worker

        mean = (0.4914, 0.4822, 0.4465)
        std  = (0.2470, 0.2435, 0.2616)
        self.normalize = T.Compose([
            T.ToTensor(),
            T.Normalize(mean, std),
        ])

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        pil_img, label = self.base[idx]    # (PIL.Image, int)

        # clean → tensor
        img_clean = self.normalize(pil_img)

        # AugMix + tensor for aug1
        img_aug1 = self.normalize(self.augmix(pil_img))

        # AugMix + tensor for aug2
        img_aug2 = self.normalize(self.augmix(pil_img))

        return img_clean, img_aug1, img_aug2, label

def create_train_val_test_ds(data_seed: int,
                             use_simple_augmix: bool = False,
                             use_advanced_augmix: bool = False,
                             augmix_config=None):
    # CIFAR-10 channel statistics
  mean = (0.4914, 0.4822, 0.4465)
  std  = (0.2470, 0.2435, 0.2616)

  assert not (use_simple_augmix and use_advanced_augmix), \
  'Only one type of AUGMIX can be active, please select one.'

  if not use_simple_augmix:
    # 1.a) “Weak” baseline augmentation for all ViT training runs
    train_transform = T.Compose([
        # zero-pad to 40×40 then random 32×32 crop → translation augmentation
        T.RandomCrop(32, padding=4),
        # 50% probability horizontal flip → reflection augmentation
        T.RandomHorizontalFlip(),
        # convert PIL image to FloatTensor in [0,1]
        T.ToTensor(),
        # subtract per-channel mean, divide by per-channel std
        T.Normalize(mean, std),
        ])

  else:
    assert augmix_config != None, "Please provide a valid configuration for AUGMIX"
    # 1.b) Standard “weak” CIFAR-10 augment + AugMix
    train_transform = T.Compose([
      T.RandomCrop(32, padding=4),
      T.RandomHorizontalFlip(),
      AugMix(**augmix_config),
      T.ToTensor(),
      T.Normalize(mean, std),
  ])

  # 2. Evaluation transforms for CIFAR-10 / CIFAR-C ->  Use for Val- and Test set
  test_transform = T.Compose([
      T.ToTensor(),
      T.Normalize(mean, std),
      ])

  # 3. Load the training set without transforms

  raw_train = datasets.CIFAR10(
      root="data",
      train=True,
      download=True,
      transform=None,       # <-------- no transform here
      target_transform=None
      )

  # 4. Load the test set with transforms

  test_dataset = datasets.CIFAR10(
      root="data",
      train=False,
      download=True,
      transform=test_transform,
  )

  # 5. Split training set into 90% train, 10% val

  total_train = len(raw_train)       # 50,000
  val_size    = total_train // 10    # 5,000
  train_size  = total_train - val_size  # 45,000

  g = torch.Generator()
  g.manual_seed(data_seed)

  train_split, val_split = random_split(dataset=raw_train,
                                    lengths=[train_size, val_size],
                                    generator=g)

  # Apply three view augmix when use_advanced_augmix is turned on
  if use_advanced_augmix:
    train_dataset = ThreeViewCIFAR_Tensor(train_split, augmix_config)
  else:
    train_dataset = WrappedSubset(train_split, transform=train_transform)

  val_dataset   = WrappedSubset(val_split, transform=test_transform)

  return train_dataset, val_dataset, test_dataset



def download_and_extract_cifar10c():
    # 1. Ensure the target directory exists
    target_dir = "data/CIFAR-10-C"
    os.makedirs(target_dir, exist_ok=True)

    # 2. Download URL and local tar‐file path
    url = "https://zenodo.org/record/2535967/files/CIFAR-10-C.tar?download=1"
    tar_path = "data/CIFAR-10-C.tar"

    # 3. Download only if not already present
    if not os.path.exists(tar_path):
        print("Downloading CIFAR-10-C (this may take a minute)...")
        urllib.request.urlretrieve(url, tar_path)
        print("Download complete.")
    else:
        print("CIFAR-10-C.tar already exists; skipping download.")

    # 4. Extract into data/CIFAR-10-C/
    print(f" Extracting {tar_path} into {target_dir}/ ...")
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=target_dir)
    print("Extraction complete.")

if __name__ == "__main__":
    # download_and_extract_cifar10c()
    pass