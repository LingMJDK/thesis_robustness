import numpy as np
from PIL import Image
import augmentations
# CIFAR-10 constants (make sure they match the correct “mean”/“std”)

# MEAN = np.array([0.4914, 0.4822, 0.4465])
# STD  = np.array([0.2470, 0.2435, 0.2616])

def normalize_np(image: np.ndarray, 
                 mean = (0.4914, 0.4822, 0.4465),
                 std  = (0.2470, 0.2435, 0.2616)
                 ) -> np.ndarray:
    """
    Normalize a NumPy image (H×W×C float32 in [0..1]) by channel‐wise mean/std.
    Returns a normalized float32 image in the same shape.
    """
    # image is H×W×C, values in [0..1]
    image = image.transpose(2, 0, 1)  # → C×H×W
    image = (image - np.array(mean)[:, None, None]) / np.array(std)[:, None, None]
    return image.transpose(1, 2, 0)   # → H×W×C

def apply_op(image: np.ndarray, op, severity: int) -> np.ndarray:
    """
    Given a float‐32 NumPy image in [0..1], apply one PIL‐based op from
    `augmentations.augmentations` (wrapped to accept (PIL, severity)) and return
    a new float‐32 NumPy image in [0..1].
    """
    # Convert to uint8 PIL
    arr255 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(arr255)        # PIL in “RGB” mode
    pil_out = op(pil_img, severity)          # op returns a PIL.Image
    arr_aug = np.asarray(pil_out).astype(np.float32) / 255.0
    return arr_aug

def augment_and_mix(
    image: np.ndarray,
    severity: int = 3,
    width: int = 3,
    depth: int = -1,
    alpha: float = 1.0,
    mean: tuple = (0.4914, 0.4822, 0.4465),
    std: tuple = (0.2470, 0.2435, 0.2616)
) -> np.ndarray:
    """
    Given a raw H×W×C NumPy image in [0..1], produce an AugMix‐augmented image
    (also H×W×C float32). You can tune `severity`, `width`, `depth`, and `alpha`.
    """
    mean = np.array(mean)
    std  = np.array(std)
    ws = np.float32(np.random.dirichlet([alpha] * width))
    m  = np.float32(np.random.beta(alpha, alpha))

    mix = np.zeros_like(image, dtype=np.float32)
    for i in range(width):
        image_aug = image.copy()
        d = depth if (depth > 0) else np.random.randint(1, 4)
        for _ in range(d):
            op = np.random.choice(augmentations.augmentations)
            image_aug = apply_op(image_aug, op, severity)

        mix += ws[i] * normalize_np(image_aug, mean, std)

    mixed = (1.0 - m) * normalize_np(image, mean, std) + m * mix
    return mixed  # still H×W×C float32

class AugMix(object):
    """
    A PyTorch‐style transform that takes a PIL image (H×W in [0..255]),
    converts it to NumPy, runs augment_and_mix, then returns a PIL image
    or Torch tensor. We’ll return a *PIL* first so that downstream T.ToTensor()
    and T.Normalize() can do their job.
    """
    def __init__(
        self,
        severity: int = 3,
        width: int = 3,
        depth: int = -1,
        alpha: float = 1.0,
        mean: tuple = (0.4914, 0.4822, 0.4465),
        std: tuple = (0.2470, 0.2435, 0.2616)
    ):
        self.severity = severity
        self.width = width
        self.depth = depth
        self.alpha = alpha
        self.mean = np.array(mean)
        self.std  = np.array(std)

    def __call__(self, pil_img):
        """
        Input: `pil_img` is a PIL.Image in RGB, with pixel‐values [0..255].
        Output: a new PIL.Image (RGB, [0..255]) representing AugMix‐normalized.
        We do:
          PIL→NumPy [0..1] → augment_and_mix(…) → denormalize back to [0..255] PIL.
        """
        # Convert PIL → float32 NumPy [0..1]
        np_img = np.asarray(pil_img).astype(np.float32) / 255.0

        # Run AugMix (gives H×W×C float32 already normalized by MEAN/STD)
        augmixed = augment_and_mix(
            np_img,
            severity=self.severity,
            width=self.width,
            depth=self.depth,
            alpha=self.alpha,
            mean=self.mean,
            std=self.std
        )

        # Convert back to [0..255] range to get a PIL.Image again
        # First “un-normalize” from (img − mean)/std → img in [0..1]
        unnorm = (augmixed * self.std[None, None, :] + self.mean[None, None, :])
        unnorm = np.clip(unnorm, 0.0, 1.0)

        # Finally, to [0..255] uint8
        arr255 = (unnorm * 255.0).astype(np.uint8)
        return Image.fromarray(arr255)
