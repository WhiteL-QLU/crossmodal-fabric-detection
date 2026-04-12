import os
import torch
import torch.fft
from PIL import Image
from torchvision import transforms
import glob
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from utils.general_utils import SquarePad

RGB_SIZE = 224


class BaseAnomalyDetectionDataset(Dataset):
    def __init__(self, split, class_name, img_size, dataset_path):
        self.IMAGENET_MEAN = [0.485, 0.456, 0.406]
        self.IMAGENET_STD = [0.229, 0.224, 0.225]
        self.cls = class_name
        self.size = img_size
        self.img_path = os.path.join(dataset_path, self.cls, split)
        self.rgb_transform = transforms.Compose([
            SquarePad(),
            transforms.Resize((RGB_SIZE, RGB_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
        ])


class TrainValDataset(BaseAnomalyDetectionDataset):
    def __init__(self, split, class_name, img_size, dataset_path):
        super().__init__(split=split, class_name=class_name, img_size=img_size, dataset_path=dataset_path)
        self.img_paths, self.labels = self.load_dataset()

    def load_dataset(self):
        rgb_paths = glob.glob(os.path.join(self.img_path, 'good') + "/*.png")
        rgb_paths.sort()
        return rgb_paths, [0] * len(rgb_paths)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        rgb_path = self.img_paths[idx]
        label = self.labels[idx]
        img = Image.open(rgb_path).convert('RGB')
        img = self.rgb_transform(img)

        gray_img = img.mean(dim=0, keepdim=True)
        fft_complex = torch.fft.fft2(gray_img)
        fft_shifted = torch.fft.fftshift(fft_complex)

        magnitude = torch.abs(fft_shifted)
        magnitude = torch.log(magnitude + 1e-8)

        phase = torch.angle(fft_shifted)
        phase = (phase + torch.pi) / (2 * torch.pi)

        fft_3channel = torch.cat([magnitude, phase, magnitude], dim=0).float()

        return (img, fft_3channel), label


class TestDataset(BaseAnomalyDetectionDataset):
    def __init__(self, class_name, img_size, dataset_path):
        super().__init__(split="test", class_name=class_name, img_size=img_size, dataset_path=dataset_path)
        self.gt_transform = transforms.Compose([
            SquarePad(),
            transforms.Resize((RGB_SIZE, RGB_SIZE), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()])
        self.img_paths, self.gt_paths, self.labels = self.load_dataset()

    def load_dataset(self):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                rgb_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                rgb_paths.sort()
                img_tot_paths.extend(rgb_paths)
                gt_tot_paths.extend([0] * len(rgb_paths))
                tot_labels.extend([0] * len(rgb_paths))
            else:
                rgb_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                gt_dir = self.img_path.replace('test', 'ground_truth')
                rgb_paths.sort()

                for rgb_path in rgb_paths:
                    img_name = os.path.basename(rgb_path)
                    gt_name = img_name.replace('.png', '_mask.png')
                    gt_path = os.path.join(gt_dir, defect_type, gt_name)

                    if not os.path.exists(gt_path):
                        gt_path = os.path.join(gt_dir, defect_type, img_name)

                    img_tot_paths.append(rgb_path)
                    gt_tot_paths.append(gt_path)
                    tot_labels.append(1)

        return img_tot_paths, gt_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        rgb_path, gt, label = self.img_paths[idx], self.gt_paths[idx], self.labels[idx]
        img_original = Image.open(rgb_path).convert('RGB')
        img = self.rgb_transform(img_original)

        gray_img = img.mean(dim=0, keepdim=True)
        fft_complex = torch.fft.fft2(gray_img)
        fft_shifted = torch.fft.fftshift(fft_complex)

        magnitude = torch.abs(fft_shifted)
        magnitude = torch.log(magnitude + 1e-8)

        phase = torch.angle(fft_shifted)
        phase = (phase + torch.pi) / (2 * torch.pi)

        fft_3channel = torch.cat([magnitude, phase, magnitude], dim=0).float()

        if gt == 0:
            gt = torch.zeros([1, self.size, self.size])
        else:
            gt = Image.open(gt).convert('L')
            gt = self.gt_transform(gt)
            gt = torch.where(gt > 0.001, 1., .0)

        return (img, fft_3channel), gt[:1], label, rgb_path


def _get_aitex_target_folder_and_normal_paths(dataset_root):
    normal_folders = sorted(os.listdir(os.path.join(dataset_root, 'NODefect_images')))
    target_folder = normal_folders[0]
    normal_paths = glob.glob(os.path.join(dataset_root, 'NODefect_images', target_folder, '*.png'))
    normal_paths.sort()
    return target_folder, normal_paths


def _get_aitex_normal_split(normal_paths, split):
    train_end = int(len(normal_paths) * 0.8)
    validation_end = int(len(normal_paths) * 0.9)

    if split == "train":
        return normal_paths[:train_end]
    if split == "validation":
        return normal_paths[train_end:validation_end]
    if split == "test":
        return normal_paths[validation_end:]

    raise ValueError(f"Unsupported AITEX split: {split}")


def _get_aitex_defect_samples(dataset_root):
    defect_dir = os.path.join(dataset_root, 'Defect_images')
    mask_dir = os.path.join(dataset_root, 'Mask_images')
    defect_paths = glob.glob(os.path.join(defect_dir, "*.png"))
    defect_paths.sort()

    defect_samples = []
    for defect_path in defect_paths:
        img_name = os.path.basename(defect_path)
        mask_name = img_name.replace('.png', '_mask.png')
        mask_path = os.path.join(mask_dir, mask_name)

        if os.path.exists(mask_path):
            defect_samples.append((defect_path, mask_path))

    return defect_samples


def _get_aitex_defect_split(defect_samples, split):
    validation_end = int(len(defect_samples) * 0.2)

    if split == "validation":
        return defect_samples[:validation_end]
    if split == "test":
        return defect_samples[validation_end:]

    raise ValueError(f"Unsupported AITEX defect split: {split}")


def _get_aitex_split_counts(normal_paths, defect_samples):
    return {
        "train_normal": len(_get_aitex_normal_split(normal_paths, "train")),
        "validation_normal": len(_get_aitex_normal_split(normal_paths, "validation")),
        "validation_defect": len(_get_aitex_defect_split(defect_samples, "validation")),
        "test_normal": len(_get_aitex_normal_split(normal_paths, "test")),
        "test_defect": len(_get_aitex_defect_split(defect_samples, "test")),
    }


class AITEXTrainDataset(BaseAnomalyDetectionDataset):
    def __init__(self, split, img_size, dataset_path):
        super().__init__(split=split, class_name="aitex", img_size=img_size, dataset_path=dataset_path)
        self.split = split
        self.dataset_root = dataset_path
        self.img_paths, self.labels = self.load_dataset()

    def load_dataset(self):
        self.target_folder, normal_paths = _get_aitex_target_folder_and_normal_paths(self.dataset_root)
        counts = _get_aitex_split_counts(normal_paths, _get_aitex_defect_samples(self.dataset_root))
        split_paths = _get_aitex_normal_split(normal_paths, self.split)
        print(
            f"[AITEX] train normal: {counts['train_normal']}, "
            f"validation normal: {counts['validation_normal']}, "
            f"validation defect: {counts['validation_defect']}, "
            f"test normal: {counts['test_normal']}, "
            f"test defect: {counts['test_defect']}"
        )
        return split_paths, [0] * len(split_paths)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        rgb_path = self.img_paths[idx]
        label = self.labels[idx]
        img = Image.open(rgb_path).convert('RGB')
        img = self.rgb_transform(img)

        gray_img = img.mean(dim=0, keepdim=True)
        fft_complex = torch.fft.fft2(gray_img)
        fft_shifted = torch.fft.fftshift(fft_complex)

        magnitude = torch.abs(fft_shifted)
        magnitude = torch.log(magnitude + 1e-8)
        phase = torch.angle(fft_shifted)
        phase = (phase + torch.pi) / (2 * torch.pi)
        fft_3channel = torch.cat([magnitude, phase, magnitude], dim=0).float()

        return (img, fft_3channel), label


class AITEXTestDataset(BaseAnomalyDetectionDataset):
    def __init__(self, split, img_size, dataset_path):
        super().__init__(split=split, class_name="aitex", img_size=img_size, dataset_path=dataset_path)
        self.split = split
        self.dataset_root = dataset_path
        self.gt_transform = transforms.Compose([
            SquarePad(),
            transforms.Resize((RGB_SIZE, RGB_SIZE), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()])
        self.img_paths, self.gt_paths, self.labels = self.load_dataset()

    def load_dataset(self):
        self.target_folder, normal_paths = _get_aitex_target_folder_and_normal_paths(self.dataset_root)
        all_defect_samples = _get_aitex_defect_samples(self.dataset_root)
        counts = _get_aitex_split_counts(normal_paths, all_defect_samples)
        eval_normal_paths = _get_aitex_normal_split(normal_paths, self.split)
        defect_samples = _get_aitex_defect_split(all_defect_samples, self.split)
        print(
            f"[AITEX] train normal: {counts['train_normal']}, "
            f"validation normal: {counts['validation_normal']}, "
            f"validation defect: {counts['validation_defect']}, "
            f"test normal: {counts['test_normal']}, "
            f"test defect: {counts['test_defect']}"
        )

        img_tot_paths = list(eval_normal_paths)
        gt_tot_paths = [0] * len(eval_normal_paths)
        tot_labels = [0] * len(eval_normal_paths)

        for defect_path, mask_path in defect_samples:
            img_tot_paths.append(defect_path)
            gt_tot_paths.append(mask_path)
            tot_labels.append(1)

        return img_tot_paths, gt_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        rgb_path, gt, label = self.img_paths[idx], self.gt_paths[idx], self.labels[idx]
        img_original = Image.open(rgb_path).convert('RGB')
        img = self.rgb_transform(img_original)

        gray_img = img.mean(dim=0, keepdim=True)
        fft_complex = torch.fft.fft2(gray_img)
        fft_shifted = torch.fft.fftshift(fft_complex)

        magnitude = torch.abs(fft_shifted)
        magnitude = torch.log(magnitude + 1e-8)
        phase = torch.angle(fft_shifted)
        phase = (phase + torch.pi) / (2 * torch.pi)
        fft_3channel = torch.cat([magnitude, phase, magnitude], dim=0).float()

        if gt == 0:
            gt = torch.zeros([1, self.size, self.size])
        else:
            gt = Image.open(gt).convert('L')
            gt = self.gt_transform(gt)
            gt = torch.where(gt > 0.001, 1., .0)

        return (img, fft_3channel), gt[:1], label, rgb_path


def get_data_loader(split, class_name, dataset_path, img_size=224, batch_size=1, shuffle=False):
    if 'AITEX' in dataset_path:
        if split in ['train']:
            dataset = AITEXTrainDataset(split=split, img_size=img_size, dataset_path=dataset_path)
        elif split in ['validation', 'test']:
            dataset = AITEXTestDataset(split=split, img_size=img_size, dataset_path=dataset_path)
    else:
        if split in ['train', 'validation']:
            dataset = TrainValDataset(split="train", class_name=class_name, img_size=img_size, dataset_path=dataset_path)
        elif split in ['test']:
            dataset = TestDataset(class_name=class_name, img_size=img_size, dataset_path=dataset_path)

    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                             num_workers=1, drop_last=False, pin_memory=True)
    return data_loader
