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
# 🌟 导入全新的双剑合璧
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

    # 🌟 加载 V2.0 解码器
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

    print(f"\n🔥 启动 V2.0 完全体推理 | 当前类别: {args.class_name}")

    for (rgb, freq_img), gt, label, rgb_path in tqdm(test_loader, desc=f'Evaluating {args.class_name}', leave=False):
        rgb, freq_img, gt = rgb.to(device), freq_img.to(device), gt.to(device)

        with torch.no_grad():
            rgb_patch, freq_patch = feature_extractor.get_features_maps(rgb, freq_img)
            
            # Batch=1 时，补全 3D 维度以喂给 2D 卷积网络
            freq_patch_batched = freq_patch.unsqueeze(0)
            
            pred_freq = CFM_RGBtoFreq(rgb_patch)
            pred_rgb = CFM_FreqtoRGB(freq_patch_batched).squeeze(0) # 出来后再削掉

            seq_len = rgb_patch.shape[0]
            spatial_dim = int(math.sqrt(seq_len))

            score_rgb = 1 - cos_sim(pred_rgb, rgb_patch).reshape(1, 1, spatial_dim, spatial_dim)
            cos_spatial = torch.nn.functional.interpolate(score_rgb, size=(224, 224), mode='bilinear', align_corners=False).squeeze().cpu().numpy()
            
            score_freq = 1 - cos_sim(pred_freq, freq_patch).reshape(1, 1, spatial_dim, spatial_dim)
            cos_freq = torch.nn.functional.interpolate(score_freq, size=(224, 224), mode='bilinear', align_corners=False).squeeze().cpu().numpy()
            
            margin = int(224 * 0.05)
            cos_spatial[:margin, :] = 0; cos_spatial[-margin:, :] = 0
            cos_spatial[:, :margin] = 0; cos_spatial[:, -margin:] = 0
            cos_spatial = gaussian_filter(cos_spatial, sigma=2) 
            
            cos_comb = np.maximum(cos_spatial, cos_freq)
            
            gts.append(gt.squeeze().cpu().numpy()) 
            predictions.append(cos_spatial) 
            
            image_labels.append(label.cpu().numpy()) 
            pixel_labels.extend(gt.flatten().cpu().numpy()) 
            
            flat_enhanced = np.sort(cos_comb.flatten())
            k_percent = int(len(flat_enhanced) * 0.01)
            image_preds.append(flat_enhanced[-k_percent:].mean()) 
            pixel_preds.extend(cos_spatial.flatten()) 

    au_pros, _ = calculate_au_pro(gts, predictions)
    pixel_rocauc = roc_auc_score(np.stack(pixel_labels), np.stack(pixel_preds))
    image_rocauc = roc_auc_score(np.stack(image_labels), np.stack(image_preds))

    img_f1 = calculate_f1_max(np.stack(image_labels), np.stack(image_preds))
    pix_f1 = calculate_f1_max(np.stack(pixel_labels), np.stack(pixel_preds))

    print(f"\n✅ V2.0 完全体 {args.class_name} 测算完成！\nI-AUROC | I-F1 | P-AUROC | P-F1 | AUPRO@30%\n  {image_rocauc:.3f}  | {img_f1:.3f} |  {pixel_rocauc:.3f}  | {pix_f1:.3f} |   {au_pros[0]:.3f}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default='./datasets/mvtec2d', type=str)
    parser.add_argument('--class_name', default="grid", type=str)
    parser.add_argument('--checkpoint_folder', default='./checkpoints/checkpoints_CFM_mvtec', type=str)
    parser.add_argument('--epochs_no', default=50, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--seed', default=3407, type=int) 
    args = parser.parse_args()
    infer_CFM(args)