import argparse
import os
import torch
import numpy as np
import math
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score, precision_recall_curve

from models.features import MultimodalFeatures
from models.dataset import get_data_loader
from models.feature_transfer_nets import DynamicPromptSIREN, MultiScale_SpatialDecoder
from utils.metrics_utils import calculate_au_pro

def calculate_f1_max(labels, scores):
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_f1_idx = np.argmax(f1_scores)
    return f1_scores[best_f1_idx]

def set_seeds(sid=42):
    np.random.seed(sid)
    torch.manual_seed(sid)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(sid)
        torch.cuda.manual_seed_all(sid)

def infer_CFM(args):
    set_seeds(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_loader = get_data_loader("test", class_name=args.class_name, img_size=224, dataset_path=args.dataset_path)
    feature_extractor = MultimodalFeatures()

    CFM_RGBtoFreq = DynamicPromptSIREN(in_features=448, out_features=1152).to(device)
    CFM_FreqtoRGB = MultiScale_SpatialDecoder(in_features=1152, out_features=448).to(device)

    model_name = f'V2_{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs'
    CFM_RGBtoFreq.load_state_dict(torch.load(rf'{args.checkpoint_folder}/{args.class_name}/CFM_RGBtoFreq_{model_name}.pth'))
    CFM_FreqtoRGB.load_state_dict(torch.load(rf'{args.checkpoint_folder}/{args.class_name}/CFM_FreqtoRGB_{model_name}.pth'))

    CFM_RGBtoFreq.eval()
    CFM_FreqtoRGB.eval()

    predictions, gts = [], []
    image_labels, pixel_labels = [], []
    image_preds, pixel_preds = [], []
    
    cos_sim = torch.nn.CosineSimilarity(dim=-1)

    print(f"\n🔥 启动 V2.1 终极解耦推理 | 当前纹理类别: {args.class_name}")

    for (rgb, freq_img), gt, label, rgb_path in tqdm(test_loader, desc=f'Evaluating {args.class_name}', leave=False):
        rgb, freq_img, gt = rgb.to(device), freq_img.to(device), gt.to(device)

        with torch.no_grad():
            rgb_patch, freq_patch = feature_extractor.get_features_maps(rgb, freq_img)
            freq_patch_batched = freq_patch.unsqueeze(0)
            
            pred_freq = CFM_RGBtoFreq(rgb_patch)
            pred_rgb = CFM_FreqtoRGB(freq_patch_batched).squeeze(0)

            seq_len = rgb_patch.shape[0]
            spatial_dim = int(math.sqrt(seq_len))

            # 1. 提取物理空间残差图 (高精度微观定位)
            score_rgb = 1 - cos_sim(pred_rgb, rgb_patch).reshape(1, 1, spatial_dim, spatial_dim)
            cos_spatial = torch.nn.functional.interpolate(score_rgb, size=(224, 224), mode='bilinear', align_corners=False).squeeze().cpu().numpy()
            
            # 2. 提取频域残差图 (宏观失调感知)
            score_freq = 1 - cos_sim(pred_freq, freq_patch).reshape(1, 1, spatial_dim, spatial_dim)
            cos_freq = torch.nn.functional.interpolate(score_freq, size=(224, 224), mode='bilinear', align_corners=False).squeeze().cpu().numpy()
            
            margin = int(224 * 0.05)
            cos_spatial[:margin, :] = 0; cos_spatial[-margin:, :] = 0
            cos_spatial[:, :margin] = 0; cos_spatial[:, -margin:] = 0
            cos_spatial = gaussian_filter(cos_spatial, sigma=2) 
            
            gts.append(gt.squeeze().cpu().numpy()) 
            predictions.append(cos_spatial) 
            
            image_labels.append(label.cpu().numpy()) 
            pixel_labels.extend(gt.flatten().cpu().numpy()) 
            
            # 🌟 核心评价解耦逻辑
            # 微观 P-AUROC 绝对信任物理空间图
            pixel_preds.extend(cos_spatial.flatten()) 
            
            # 宏观 I-AUROC 提取：空间最严重的地方 + 频域的全局异常波动
            # 使用加权而非 crude 的 maximum
            flat_spatial = np.sort(cos_spatial.flatten())
            spatial_anomaly_score = flat_spatial[-int(len(flat_spatial) * 0.01):].mean()
            freq_anomaly_score = np.mean(cos_freq) 
            
            # 科学融合特征：空间剧烈异动，或频域全局失调，都会拉高图片异常分
            image_score = (spatial_anomaly_score * 0.6) + (freq_anomaly_score * 0.4)
            image_preds.append(image_score)

    au_pros, _ = calculate_au_pro(gts, predictions)
    pixel_rocauc = roc_auc_score(np.stack(pixel_labels), np.stack(pixel_preds))
    image_rocauc = roc_auc_score(np.stack(image_labels), np.stack(image_preds))

    img_f1 = calculate_f1_max(np.stack(image_labels), np.stack(image_preds))
    pix_f1 = calculate_f1_max(np.stack(pixel_labels), np.stack(pixel_preds))

    print(f"\n✅ V2.1 {args.class_name} 测算完成！\nI-AUROC | I-F1 | P-AUROC | P-F1 | AUPRO@30%\n  {image_rocauc:.3f}  | {img_f1:.3f} |  {pixel_rocauc:.3f}  | {pix_f1:.3f} |   {au_pros[0]:.3f}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default='./datasets/mvtec2d', type=str)
    parser.add_argument('--class_name', default="carpet", type=str)
    parser.add_argument('--checkpoint_folder', default='./checkpoints/checkpoints_CFM_mvtec', type=str)
    parser.add_argument('--epochs_no', default=50, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--seed', default=3407, type=int) 
    args = parser.parse_args()
    infer_CFM(args)