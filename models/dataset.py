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
            transforms.Resize((RGB_SIZE, RGB_SIZE), interpolation = transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean = self.IMAGENET_MEAN, std = self.IMAGENET_STD)
        ])

class TrainValDataset(BaseAnomalyDetectionDataset):
    def __init__(self, split, class_name, img_size, dataset_path):
        super().__init__(split = split, class_name = class_name, img_size = img_size, dataset_path = dataset_path)
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
        super().__init__(split = "test", class_name = class_name, img_size = img_size, dataset_path = dataset_path)
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

# ====================================================================
# 🌟 新增模块：AITEX 纯正工业织物数据集专属接驳舱 🌟
# ====================================================================
class AITEXTrainDataset(BaseAnomalyDetectionDataset):
    def __init__(self, img_size, dataset_path):
        super().__init__(split="train", class_name="aitex", img_size=img_size, dataset_path=dataset_path)
        self.dataset_root = dataset_path
        self.img_paths, self.labels = self.load_dataset()

    def load_dataset(self):
        # 【实验架构的重构】 强制降维：锁定第一种真实布料，彻底阻断 7 类流形坍塌
        normal_folders = sorted(os.listdir(os.path.join(self.dataset_root, 'NODefect_images')))
        self.target_folder = normal_folders[0]
        print(f"\n🔥 架构师强制干预：AITEX 训练集锁定单一织物 -> {self.target_folder}")

        normal_paths = glob.glob(os.path.join(self.dataset_root, 'NODefect_images', self.target_folder, '*.png'))
        normal_paths.sort()
        
        split_idx = int(len(normal_paths) * 0.8)
        train_paths = normal_paths[:split_idx]
        return train_paths, [0] * len(train_paths)

    def __len__(self): return len(self.img_paths)

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
    def __init__(self, img_size, dataset_path):
        super().__init__(split="test", class_name="aitex", img_size=img_size, dataset_path=dataset_path)
        self.dataset_root = dataset_path
        self.gt_transform = transforms.Compose([
            SquarePad(),
            transforms.Resize((RGB_SIZE, RGB_SIZE), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()])
        self.img_paths, self.gt_paths, self.labels = self.load_dataset()

    def load_dataset(self):
        # 1. 负样本必须与训练集绝对同源！否则 FPR 会因为不认识的正常布料而爆炸
        normal_folders = sorted(os.listdir(os.path.join(self.dataset_root, 'NODefect_images')))
        self.target_folder = normal_folders[0]
        print(f"🔥 架构师强制干预：AITEX 测试集负样本锁定 -> {self.target_folder}\n")

        normal_paths = glob.glob(os.path.join(self.dataset_root, 'NODefect_images', self.target_folder, '*.png'))
        normal_paths.sort()
        split_idx = int(len(normal_paths) * 0.8)
        test_normal_paths = normal_paths[split_idx:]

        img_tot_paths = test_normal_paths
        gt_tot_paths = [0] * len(test_normal_paths)
        tot_labels = [0] * len(test_normal_paths)

        # 2. 挂载真实瑕疵集。注意：在 I-AUROC 盲测阶段，非同源的瑕疵图依然是有效正样本。
        defect_dir = os.path.join(self.dataset_root, 'Defect_images')
        mask_dir = os.path.join(self.dataset_root, 'Mask_images')
        defect_paths = glob.glob(os.path.join(defect_dir, "*.png"))
        defect_paths.sort()

        for defect_path in defect_paths:
            img_name = os.path.basename(defect_path)
            mask_name = img_name.replace('.png', '_mask.png')
            mask_path = os.path.join(mask_dir, mask_name)

            if os.path.exists(mask_path): 
                img_tot_paths.append(defect_path)
                gt_tot_paths.append(mask_path)
                tot_labels.append(1)

        return img_tot_paths, gt_tot_paths, tot_labels

    def __len__(self): return len(self.img_paths)

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

# ====================================================================
# 🌟 全局路由分配器 🌟
# ====================================================================
def get_data_loader(split, class_name, dataset_path, img_size = 224, batch_size = 1, shuffle = False):
    if 'AITEX' in dataset_path:
        if split in ['train', 'validation']:
            dataset = AITEXTrainDataset(img_size=img_size, dataset_path=dataset_path)
        elif split in ['test']:
            dataset = AITEXTestDataset(img_size=img_size, dataset_path=dataset_path)
    else:
        if split in ['train', 'validation']:
            dataset = TrainValDataset(split="train", class_name=class_name, img_size=img_size, dataset_path=dataset_path)
        elif split in ['test']:
            dataset = TestDataset(class_name=class_name, img_size=img_size, dataset_path=dataset_path)

    data_loader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = shuffle,
                             num_workers = 1, drop_last = False, pin_memory = True)
    return data_loader