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
import urllib.request
import zipfile
from torchvision.datasets import STL10, ImageFolder, CIFAR10
from torch.utils.data import ConcatDataset, DataLoader
import fire

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
    def __init__(self, base_subset, 
                 augmix_config, 
                 mean= (0.4914, 0.4822, 0.4465), 
                 std=(0.2470, 0.2435, 0.2616)):
        self.base   = base_subset
        self.augmix = AugMix(**augmix_config)  # create once per worker
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

def create_train_val_test_ds(
    data_seed: int,
    use_simple_augmix: bool = False,
    use_advanced_augmix: bool = False,
    augmix_config=None,
    root: str = 'data',
    mean: tuple = (0.4914, 0.4822, 0.4465),
    std: tuple = (0.2470, 0.2435, 0.2616),
    dataset_cls=datasets.CIFAR10,      # new argument
    dataset_kwargs: dict = None                    # optional extra kwargs
):
    assert not (use_simple_augmix and use_advanced_augmix), \
        'Only one type of AUGMIX can be active, please select one.'

    if dataset_kwargs is None:
        dataset_kwargs = {}

    # 1. Build train_transform
    if not use_simple_augmix:
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    else:
        assert augmix_config is not None, "Please provide a valid configuration for AUGMIX"
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            AugMix(**augmix_config),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])

    # 2. Build test_transform
    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    # 3. Load raw train (no transform)
    raw_train = dataset_cls(
        root=root,
        train=True,
        download=True,
        transform=None,
        target_transform=None,
        **dataset_kwargs
    )

    # 4. Load test set (with transforms)
    test_dataset = dataset_cls(
        root=root,
        train=False,
        download=True,
        transform=test_transform,
        **dataset_kwargs
    )

    # 5. Split into train/val
    total = len(raw_train)
    val_size = total // 10
    train_size = total - val_size

    g = torch.Generator().manual_seed(data_seed)
    train_split, val_split = random_split(
        dataset=raw_train,
        lengths=[train_size, val_size],
        generator=g
    )

    # 6. Wrap splits
    if use_advanced_augmix:
        train_dataset = ThreeViewCIFAR_Tensor(train_split, augmix_config, mean=mean, std=std)
    else:
        train_dataset = WrappedSubset(train_split, transform=train_transform)

    val_dataset = WrappedSubset(val_split, transform=test_transform)

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



def get_tiny_imagenet_dataset(
    root: str = "data",
    train_transform=None,
    url: str = "http://cs231n.stanford.edu/tiny-imagenet-200.zip",
    archive_name: str = "tiny-imagenet-200.zip",
    extract_dir: str = "tiny-imagenet-200"
) -> ImageFolder:
    """
    Ensure TinyImageNet is downloaded+extracted under `root/extract_dir`.
    Returns an ImageFolder for the train split.

    Args:
      root           : base folder for download/extraction
      train_transform: transform to apply to training images
      url            : download URL for the zip archive
      archive_name   : local filename under `root` for the zip
      extract_dir    : subfolder name after extraction
    """
    archive_path = os.path.join(root, archive_name)
    data_dir     = os.path.join(root, extract_dir)
    train_dir    = os.path.join(data_dir, "train")

    # 1) Download & extract if the *train* folder doesn't yet exist
    if not os.path.isdir(train_dir):
        os.makedirs(root, exist_ok=True)
        if not os.path.exists(archive_path):
            print(f"Downloading TinyImageNet to {archive_path}...")
            urllib.request.urlretrieve(url, archive_path)
            print("Download complete.")
        print(f"Extracting {archive_path} → {data_dir}/")
        with zipfile.ZipFile(archive_path, "r") as z:
            z.extractall(root)
        print("Extraction complete.")

    # 2) Return the ImageFolder
    return ImageFolder(train_dir, transform=train_transform)

def create_pretrain_loaders(
    split_seed: int,
    root: str = 'data',
    image_size: int = 32,
    batch_size: int = 256,
    num_workers: int = 10,
    mean: tuple = (0.485, 0.456, 0.406),
    std: tuple  = (0.229, 0.224, 0.225),
    val_split: float = 0.1,
    prefetch_factor: int = 2,
    persistent_workers: bool = True
):
    """
    Build DataLoaders over the STL-10 unlabeled split + TinyImageNet train split + CIFAR-10 train split,
    all resized to `image_size` and normalized by the given mean/std, then split
    into train & val for MAE pretraining.

    Returns:
      train_loader, val_loader
    """
    import torchvision.transforms as T
    from torchvision.datasets import STL10, ImageFolder, CIFAR10
    from torch.utils.data import ConcatDataset, DataLoader, Subset
    import os
    import torch

    tf_train = T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomCrop(image_size, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    tf_val = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    # Helper to split indices
    def split_indices(n, val_split, seed):
        g = torch.Generator().manual_seed(seed)
        idxs = torch.randperm(n, generator=g)
        val_size = int(val_split * n)
        return idxs[val_size:], idxs[:val_size]

    # STL10
    stl_full = STL10(root, split='unlabeled', download=True)
    stl_train_idx, stl_val_idx = split_indices(len(stl_full), val_split, split_seed)
    stl_train = Subset(STL10(root, split='unlabeled', download=True, transform=tf_train), stl_train_idx)
    stl_val   = Subset(STL10(root, split='unlabeled', download=True, transform=tf_val), stl_val_idx)

    # TinyImageNet
    tin_full = ImageFolder(os.path.join(root, 'tiny-imagenet-200', 'train'))
    tin_train_idx, tin_val_idx = split_indices(len(tin_full), val_split, split_seed)
    tin_train = Subset(ImageFolder(os.path.join(root, 'tiny-imagenet-200', 'train'), transform=tf_train), tin_train_idx)
    tin_val   = Subset(ImageFolder(os.path.join(root, 'tiny-imagenet-200', 'train'), transform=tf_val), tin_val_idx)

    # CIFAR10
    cifar_full = CIFAR10(root, train=True, download=True)
    cifar_train_idx, cifar_val_idx = split_indices(len(cifar_full), val_split, split_seed)
    cifar_train = Subset(CIFAR10(root, train=True, download=True, transform=tf_train), cifar_train_idx)
    cifar_val   = Subset(CIFAR10(root, train=True, download=True, transform=tf_val), cifar_val_idx)

    # Concat
    train_ds = ConcatDataset([stl_train, tin_train, cifar_train])
    val_ds   = ConcatDataset([stl_val, tin_val, cifar_val])

    train_loader = DataLoader(train_ds,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True,
                              prefetch_factor=prefetch_factor,
                              persistent_workers=persistent_workers
    )
    val_loader = DataLoader(val_ds,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True,
                            prefetch_factor=prefetch_factor,
                            persistent_workers=persistent_workers
    )

    return train_loader, val_loader
# def create_pretrain_loaders(
#     split_seed: int,
#     root: str = 'data',
#     image_size: int = 32,  # changed default to 32
#     batch_size: int = 256,
#     num_workers: int = 10,
#     mean: tuple = (0.485, 0.456, 0.406),
#     std: tuple  = (0.229, 0.224, 0.225),
#     val_split: float = 0.1,
#     prefetch_factor: int = 2,
#     persistent_workers: bool = True

# ):
#     """
#     Build DataLoaders over the STL-10 unlabeled split + TinyImageNet train split + CIFAR-10 train split,
#     all resized to `image_size` and normalized by the given mean/std, then split
#     into train & val for MAE pretraining.

#     Returns:
#       train_loader, val_loader
#     """
#     tf = T.Compose([
#         T.Resize((image_size, image_size)),
#         T.RandomCrop(image_size, padding=4),
#         T.RandomHorizontalFlip(),
#         T.ToTensor(),
#         T.Normalize(mean, std),
#     ])
    
    

#     # 1) Datasets
#     stl = STL10(root, split='unlabeled', download=True, transform=tf)
#     tin = ImageFolder(os.path.join(root, 'tiny-imagenet-200', 'train'), transform=tf)
#     cifar = CIFAR10(root, train=True, download=True, transform=tf)

#     full = ConcatDataset([stl, tin, cifar])

#     # 2) split
#     total = len(full)
#     val_size = int(val_split * total)
#     train_size = total - val_size
#     g = torch.Generator().manual_seed(split_seed)
#     train_ds, val_ds = random_split(full, [train_size, val_size], generator=g)

#     # 3) loaders
#     train_loader = DataLoader(train_ds,
#                               batch_size=batch_size,
#                               shuffle=True,
#                               num_workers=num_workers,
#                               pin_memory=True,
#                               prefetch_factor=prefetch_factor,
#                               persistent_workers=persistent_workers
#     )
#     val_loader = DataLoader(val_ds,
#                             batch_size=batch_size,
#                             shuffle=False,
#                             num_workers=num_workers,
#                             pin_memory=True,
#                             prefetch_factor=prefetch_factor,
#                             persistent_workers=persistent_workers
#     )

#     return train_loader, val_loader


if __name__ == "__main__":
    # download_and_extract_cifar10c()
    fire.Fire(get_tiny_imagenet_dataset)
    pass