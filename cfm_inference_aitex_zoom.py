import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score, precision_recall_curve
import torch.nn.functional as F

from models.features import MultimodalFeatures
from models.dataset import get_data_loader
from models.feature_transfer_nets import DynamicPromptSIREN
from utils.metrics_utils import calculate_au_pro

def calculate_f1_max(labels, scores):
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_f1_idx = np.argmax(f1_scores)
    return f1_scores[best_f1_idx]

def set_seeds(sid=42):
    np.random.seed(sid)
    torch.manual_seed(sid)
    torch.cuda.manual_seed_all(sid)

def infer_CFM_AITEX_Zoom(args):
    set_seeds(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 强制 batch_size=1，因为我们要对单张图进行切片拼接
    test_loader = get_data_loader("test", class_name=args.class_name, img_size=224, dataset_path=args.dataset_path, batch_size=1)
    feature_extractor = MultimodalFeatures()

    CFM_RGBtoFreq = DynamicPromptSIREN(in_features=768, out_features=1152).to(device)
    CFM_FreqtoRGB = DynamicPromptSIREN(in_features=1152, out_features=768).to(device)

    model_name = f'{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs'
    CFM_RGBtoFreq.load_state_dict(torch.load(rf'{args.checkpoint_folder}/{args.class_name}/CFM_RGBtoFreq_{model_name}.pth'))
    CFM_FreqtoRGB.load_state_dict(torch.load(rf'{args.checkpoint_folder}/{args.class_name}/CFM_FreqtoRGB_{model_name}.pth'))

    CFM_RGBtoFreq.eval()
    CFM_FreqtoRGB.eval()

    predictions, gts = [], []
    image_labels, pixel_labels = [], []
    image_preds, pixel_preds = [], []
    cos_sim = torch.nn.CosineSimilarity(dim=1)

    for (rgb, freq_img), gt, label, rgb_path in tqdm(test_loader, desc=f'Evaluating AITEX (Zoom-in)', leave=False):
        gt_np = gt.squeeze().cpu().numpy()
        
        with torch.no_grad():
            # 🌟 创新点：Test-Time Zoom-in (无需重新训练)
            # 我们将 224x224 的原图，物理放大 2 倍到 448x448
            # 然后切成 4 块 224x224 的子图分别送入网络，强制网络当显微镜用
            rgb_large = F.interpolate(rgb, size=(448, 448), mode='bilinear', align_corners=False)
            freq_large = F.interpolate(freq_img, size=(448, 448), mode='bilinear', align_corners=False)
            
            # 裁剪坐标 (top, left, bottom, right)
            crops = [
                (0, 0, 224, 224),       # 左上
                (0, 224, 224, 448),     # 右上
                (224, 0, 448, 224),     # 左下
                (224, 224, 448, 448)    # 右下
            ]
            
            heatmap_large = np.zeros((448, 448))
            
            for (y1, x1, y2, x2) in crops:
                rgb_crop = rgb_large[:, :, y1:y2, x1:x2].to(device)
                freq_crop = freq_large[:, :, y1:y2, x1:x2].to(device)
                
                rgb_patch, freq_patch = feature_extractor.get_features_maps(rgb_crop, freq_crop)
                pred_rgb = CFM_FreqtoRGB(freq_patch)
                pred_freq = CFM_RGBtoFreq(rgb_patch)
                
                # 图像级报警依然使用双流
                score_rgb = 1 - cos_sim(pred_rgb, rgb_patch).reshape(1, 1, 28, 28)
                score_freq = 1 - cos_sim(pred_freq, freq_patch).reshape(1, 1, 28, 28)
                
                cos_s = F.interpolate(score_rgb, size=(224, 224), mode='bilinear', align_corners=False).squeeze().cpu().numpy()
                cos_f = F.interpolate(score_freq, size=(224, 224), mode='bilinear', align_corners=False).squeeze().cpu().numpy()
                
                cos_comb = np.maximum(cos_s, cos_f)
                
                # 边缘截断
                margin = int(224 * 0.05)
                cos_comb[:margin, :] = 0; cos_comb[-margin:, :] = 0
                cos_comb[:, :margin] = 0; cos_comb[:, -margin:] = 0
                
                # 拼图
                heatmap_large[y1:y2, x1:x2] = cos_comb

            # 将拼装好的 448x448 巨幅热力图，缩放回原始的 224x224
            heatmap_large_tensor = torch.tensor(heatmap_large).unsqueeze(0).unsqueeze(0)
            final_heatmap = F.interpolate(heatmap_large_tensor, size=(224, 224), mode='bilinear', align_corners=False).squeeze().numpy()
            
            # 极小的高斯模糊，保护细小像素
            final_heatmap = gaussian_filter(final_heatmap, sigma=1)
            
            gts.append(gt_np) 
            predictions.append(final_heatmap) 
            
            image_labels.append(label.cpu().numpy()) 
            pixel_labels.extend(gt_np.flatten()) 
            
            flat_enhanced = np.sort(final_heatmap.flatten())
            k_percent = int(len(flat_enhanced) * 0.01)
            image_preds.append(flat_enhanced[-k_percent:].mean()) 
            pixel_preds.extend(final_heatmap.flatten()) 

    au_pros, _ = calculate_au_pro(gts, predictions)
    pixel_rocauc = roc_auc_score(np.stack(pixel_labels), np.stack(pixel_preds))
    image_rocauc = roc_auc_score(np.stack(image_labels), np.stack(image_preds))

    img_f1 = calculate_f1_max(np.stack(image_labels), np.stack(image_preds))
    pix_f1 = calculate_f1_max(np.stack(pixel_labels), np.stack(pixel_preds))

    print(f"\n✅ AITEX 局部显微镜放大推理完成！\nI-AUROC: {image_rocauc:.3f} | I-F1: {img_f1:.3f} | P-AUROC: {pixel_rocauc:.3f} | P-F1: {pix_f1:.3f} | AUPRO: {au_pros[0]:.3f}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default='./datasets/AITEX', type=str)
    parser.add_argument('--class_name', default="aitex", type=str)
    parser.add_argument('--checkpoint_folder', default='./checkpoints/checkpoints_CFM_mvtec', type=str)
    parser.add_argument('--epochs_no', default=50, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--seed', default=3407, type=int) 
    args = parser.parse_args()
    infer_CFM_AITEX_Zoom(args)