import glob
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from utils.general_utils import SquarePad

RGB_SIZE = 224


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_fft_features(img_tensor):
    gray_img = img_tensor.mean(dim=0, keepdim=True)
    fft_complex = torch.fft.fft2(gray_img)
    fft_shifted = torch.fft.fftshift(fft_complex)

    magnitude = torch.abs(fft_shifted)
    magnitude = torch.log(magnitude + 1e-8)

    phase = torch.angle(fft_shifted)
    phase = (phase + torch.pi) / (2 * torch.pi)

    return torch.cat([magnitude, phase, magnitude], dim=0).float()


class BaseSegmentationDataset(Dataset):
    def __init__(self, class_name, img_size, dataset_path):
        self.class_name = class_name
        self.size = img_size
        self.dataset_root = dataset_path

        self.imagenet_mean = [0.485, 0.456, 0.406]
        self.imagenet_std = [0.229, 0.224, 0.225]

        self.rgb_transform = transforms.Compose([
            SquarePad(),
            transforms.Resize((RGB_SIZE, RGB_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.imagenet_mean, std=self.imagenet_std),
        ])

        self.gt_transform = transforms.Compose([
            SquarePad(),
            transforms.Resize((RGB_SIZE, RGB_SIZE), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])


def _resolve_target_normal_folder(dataset_root, class_name):
    normal_root = os.path.join(dataset_root, "NODefect_images")
    available_folders = sorted(
        folder_name
        for folder_name in os.listdir(normal_root)
        if os.path.isdir(os.path.join(normal_root, folder_name))
    )

    if not available_folders:
        raise FileNotFoundError(f"No normal folders found under: {normal_root}")

    if class_name in available_folders:
        return class_name

    lower_to_name = {folder_name.lower(): folder_name for folder_name in available_folders}
    if class_name.lower() in lower_to_name:
        return lower_to_name[class_name.lower()]

    if class_name.lower() == "aitex" or len(available_folders) == 1:
        return available_folders[0]

    raise ValueError(
        f"class_name '{class_name}' does not match any AITEX normal folder. "
        f"Available folders: {available_folders}"
    )


def _get_normal_paths(dataset_root, class_name):
    target_folder = _resolve_target_normal_folder(dataset_root, class_name)
    normal_paths = glob.glob(os.path.join(dataset_root, "NODefect_images", target_folder, "*.png"))
    normal_paths.sort()
    return target_folder, normal_paths


def _split_normal_paths(normal_paths, split):
    train_end = int(len(normal_paths) * 0.8)
    validation_end = int(len(normal_paths) * 0.9)

    if split == "train":
        return normal_paths[:train_end]
    if split == "validation":
        return normal_paths[train_end:validation_end]
    if split == "test":
        return normal_paths[validation_end:]

    raise ValueError(f"Unsupported split: {split}")


def _find_mask_path(mask_dir, defect_path):
    image_name = os.path.basename(defect_path)
    default_mask_path = os.path.join(mask_dir, image_name.replace(".png", "_mask.png"))
    if os.path.exists(default_mask_path):
        return default_mask_path

    fallback_mask_path = os.path.join(mask_dir, image_name)
    if os.path.exists(fallback_mask_path):
        return fallback_mask_path

    return None


def _get_defect_samples(dataset_root):
    defect_dir = os.path.join(dataset_root, "Defect_images")
    mask_dir = os.path.join(dataset_root, "Mask_images")
    defect_paths = glob.glob(os.path.join(defect_dir, "*.png"))
    defect_paths.sort()

    defect_samples = []
    for defect_path in defect_paths:
        mask_path = _find_mask_path(mask_dir, defect_path)
        if mask_path is not None:
            defect_samples.append((defect_path, mask_path))

    return defect_samples


def _split_defect_samples(defect_samples, split):
    train_end = int(len(defect_samples) * 0.6)
    validation_end = int(len(defect_samples) * 0.8)

    if split == "train":
        return defect_samples[:train_end]
    if split == "validation":
        return defect_samples[train_end:validation_end]
    if split == "test":
        return defect_samples[validation_end:]

    raise ValueError(f"Unsupported split: {split}")


class AITEXSegDataset(BaseSegmentationDataset):
    def __init__(self, split, class_name, img_size, dataset_path):
        super().__init__(class_name=class_name, img_size=img_size, dataset_path=dataset_path)
        self.split = split
        self.target_folder, self.normal_paths = _get_normal_paths(self.dataset_root, self.class_name)
        self.defect_samples = _get_defect_samples(self.dataset_root)
        self.samples = self._build_samples()

    def _build_samples(self):
        split_normal_paths = _split_normal_paths(self.normal_paths, self.split)
        split_defect_samples = _split_defect_samples(self.defect_samples, self.split)

        samples = [(rgb_path, None, 0) for rgb_path in split_normal_paths]
        samples.extend((rgb_path, mask_path, 1) for rgb_path, mask_path in split_defect_samples)

        print(
            f"[AITEX-Seg] folder={self.target_folder} | split={self.split} | "
            f"normal={len(split_normal_paths)} | defect={len(split_defect_samples)}"
        )
        return samples

    def __len__(self):
        return len(self.samples)

    def _load_mask(self, mask_path):
        if mask_path is None:
            return torch.zeros((1, self.size, self.size), dtype=torch.float32)

        mask = Image.open(mask_path).convert("L")
        mask = self.gt_transform(mask)
        mask = torch.where(mask > 0.001, 1.0, 0.0)
        return mask[:1].float()

    def __getitem__(self, idx):
        rgb_path, mask_path, label = self.samples[idx]
        img = Image.open(rgb_path).convert("RGB")
        img = self.rgb_transform(img)
        fft_3channel = build_fft_features(img)
        gt_mask = self._load_mask(mask_path)

        return (img, fft_3channel), gt_mask, label, rgb_path


def get_seg_data_loader(
    split,
    class_name,
    dataset_path,
    img_size=224,
    batch_size=1,
    shuffle=False,
    seed=None,
    num_workers=1,
):
    dataset = AITEXSegDataset(
        split=split,
        class_name=class_name,
        img_size=img_size,
        dataset_path=dataset_path,
    )

    generator = None
    worker_init_fn = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
        worker_init_fn = seed_worker

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init_fn,
        generator=generator,
    )
