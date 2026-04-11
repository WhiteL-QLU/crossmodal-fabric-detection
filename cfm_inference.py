import argparse
import os
import torch
import numpy as np
import math
import cv2
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

def get_postprocess_config(variant):
    configs = {
        "A": {
            "use_border_suppression": True,
            "use_gaussian_smoothing": True,
            "description": "border suppression + gaussian_filter(sigma=2)",
        },
        "B": {
            "use_border_suppression": False,
            "use_gaussian_smoothing": True,
            "description": "gaussian_filter(sigma=2) only",
        },
        "C": {
            "use_border_suppression": True,
            "use_gaussian_smoothing": False,
            "description": "border suppression only",
        },
        "D": {
            "use_border_suppression": False,
            "use_gaussian_smoothing": False,
            "description": "no border suppression and no gaussian smoothing",
        },
    }
    return configs[variant.upper()]

def apply_post_processing(anomaly_map, variant, border_ratio=0.05, gaussian_sigma=2):
    processed_map = anomaly_map.copy()
    config = get_postprocess_config(variant)

    if config["use_border_suppression"]:
        margin_h = int(processed_map.shape[0] * border_ratio)
        margin_w = int(processed_map.shape[1] * border_ratio)
        if margin_h > 0:
            processed_map[:margin_h, :] = 0
            processed_map[-margin_h:, :] = 0
        if margin_w > 0:
            processed_map[:, :margin_w] = 0
            processed_map[:, -margin_w:] = 0

    if config["use_gaussian_smoothing"]:
        processed_map = gaussian_filter(processed_map, sigma=gaussian_sigma)

    return processed_map

def save_anomaly_heatmap(anomaly_map, rgb_path, output_dir, class_name, variant, save_index):
    if isinstance(rgb_path, (list, tuple)):
        rgb_path = rgb_path[0]

    sample_name = os.path.splitext(os.path.basename(rgb_path))[0]
    defect_name = os.path.basename(os.path.dirname(rgb_path))

    normalized_map = anomaly_map - anomaly_map.min()
    if normalized_map.max() > 0:
        normalized_map = normalized_map / normalized_map.max()
    heatmap_uint8 = np.uint8(normalized_map * 255.0)
    heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    save_folder = os.path.join(output_dir, class_name, f"variant_{variant}")
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, f"{save_index:02d}_{defect_name}_{sample_name}.png")
    cv2.imwrite(save_path, heatmap_bgr)

def infer_CFM(args):
    set_seeds(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    postprocess_config = get_postprocess_config(args.postproc_variant)
    saved_heatmaps = 0

    test_loader = get_data_loader("test", class_name=args.class_name, img_size=224, dataset_path=args.dataset_path)
    print(f"[DEBUG] test_loader len: {len(test_loader)}")
    feature_extractor = MultimodalFeatures()

    CFM_RGBtoFreq = DynamicPromptSIREN(in_features=448, out_features=1152).to(device)
    CFM_FreqtoRGB = MultiScale_SpatialDecoder(in_features=1152, out_features=448).to(device)

    model_name = f'V2_Unfrozen_{args.class_name}_{args.epochs_no}ep'
    CFM_RGBtoFreq.load_state_dict(torch.load(rf'{args.checkpoint_folder}/{args.class_name}/CFM_RGBtoFreq_{model_name}.pth'))
    CFM_FreqtoRGB.load_state_dict(torch.load(rf'{args.checkpoint_folder}/{args.class_name}/CFM_FreqtoRGB_{model_name}.pth'))

    CFM_RGBtoFreq.eval()
    CFM_FreqtoRGB.eval()

    predictions, gts = [], []
    image_labels, pixel_labels = [], []
    image_preds, pixel_preds = [], []
    first_batch_debug_printed = False
    first_patch_debug_printed = False
    
    cos_sim = torch.nn.CosineSimilarity(dim=-1)

    print(f"\n馃敟 鍚姩 V2.1 缁堟瀬瑙ｈ€︽帹鐞?| 褰撳墠绾圭悊绫诲埆: {args.class_name}")

    print(f"Post-processing variant {args.postproc_variant}: {postprocess_config['description']}")
    for (rgb, freq_img), gt, label, rgb_path in tqdm(test_loader, desc=f'Evaluating {args.class_name}', leave=False):
        if not first_batch_debug_printed:
            sample_path = rgb_path[0] if isinstance(rgb_path, (list, tuple)) and len(rgb_path) > 0 else rgb_path
            label_value = label.view(-1).detach().cpu().tolist() if torch.is_tensor(label) else label
            print("[DEBUG] first batch:")
            print(f"  rgb: {tuple(rgb.shape)}")
            print(f"  gt: {tuple(gt.shape)}")
            print(f"  label: shape={tuple(label.shape)} value={label_value}")
            print(f"  path: {sample_path}")
            first_batch_debug_printed = True

        rgb, freq_img, gt = rgb.to(device), freq_img.to(device), gt.to(device)

        with torch.no_grad():
            rgb_patch, freq_patch = feature_extractor.get_features_maps(rgb, freq_img)
            if not first_patch_debug_printed:
                debug_seq_len = rgb_patch.shape[0] if rgb_patch.dim() == 2 else rgb_patch.shape[1]
                debug_spatial_dim = int(math.sqrt(debug_seq_len))
                print(f"[DEBUG] rgb_patch: {tuple(rgb_patch.shape)}")
                print(f"[DEBUG] freq_patch: {tuple(freq_patch.shape)}")
                print(f"[DEBUG] seq_len: {debug_seq_len}")
                print(f"[DEBUG] spatial_dim: {debug_spatial_dim}")
                first_patch_debug_printed = True
            freq_patch_batched = freq_patch.unsqueeze(0)
            
            pred_freq = CFM_RGBtoFreq(rgb_patch)
            pred_rgb = CFM_FreqtoRGB(freq_patch_batched).squeeze(0)

            seq_len = rgb_patch.shape[0]
            spatial_dim = int(math.sqrt(seq_len))

            # 1. 鎻愬彇鐗╃悊绌洪棿娈嬪樊鍥?(楂樼簿搴﹀井瑙傚畾浣?
            score_rgb = 1 - cos_sim(pred_rgb, rgb_patch).reshape(1, 1, spatial_dim, spatial_dim)
            cos_spatial = torch.nn.functional.interpolate(score_rgb, size=(224, 224), mode='bilinear', align_corners=False).squeeze().cpu().numpy()
            
            # 2. 鎻愬彇棰戝煙娈嬪樊鍥?(瀹忚澶辫皟鎰熺煡)
            score_freq = 1 - cos_sim(pred_freq, freq_patch).reshape(1, 1, spatial_dim, spatial_dim)
            cos_freq = torch.nn.functional.interpolate(score_freq, size=(224, 224), mode='bilinear', align_corners=False).squeeze().cpu().numpy()
            
            cos_spatial = apply_post_processing(cos_spatial, args.postproc_variant)
            
            gts.append(gt.squeeze().cpu().numpy()) 
            predictions.append(cos_spatial) 
            
            image_labels.append(label.cpu().numpy()) 
            pixel_labels.extend(gt.flatten().cpu().numpy()) 
            
            label_value = int(label.view(-1)[0].item())
            if label_value == 1 and saved_heatmaps < args.max_heatmaps:
                save_anomaly_heatmap(
                    cos_spatial,
                    rgb_path,
                    args.heatmap_dir,
                    args.class_name,
                    args.postproc_variant,
                    saved_heatmaps,
                )
                saved_heatmaps += 1
            
            # 馃専 鏍稿績璇勪环瑙ｈ€﹂€昏緫
            # 微观 P-AUROC 绝对信任物理空间图
            pixel_preds.extend(cos_spatial.flatten())
            
            # 瀹忚 I-AUROC 鎻愬彇锛氱┖闂存渶涓ラ噸鐨勫湴鏂?+ 棰戝煙鐨勫叏灞€寮傚父娉㈠姩
            # 浣跨敤鍔犳潈鑰岄潪 crude 鐨?maximum
            flat_spatial = np.sort(cos_spatial.flatten())
            spatial_anomaly_score = flat_spatial[-int(len(flat_spatial) * 0.01):].mean()
            freq_anomaly_score = np.mean(cos_freq) 
            
            # 绉戝铻嶅悎鐗瑰緛锛氱┖闂村墽鐑堝紓鍔紝鎴栭鍩熷叏灞€澶辫皟锛岄兘浼氭媺楂樺浘鐗囧紓甯稿垎
            image_score = (spatial_anomaly_score * 0.6) + (freq_anomaly_score * 0.4)
            image_preds.append(image_score)

    debug_lengths = {
        "pixel_labels": len(pixel_labels),
        "pixel_preds": len(pixel_preds),
        "image_labels": len(image_labels),
        "image_preds": len(image_preds),
        "gts": len(gts),
        "predictions": len(predictions),
    }
    print("[DEBUG] before metrics:")
    for name, length in debug_lengths.items():
        print(f"  {name}: {length}")
    empty_items = [name for name, length in debug_lengths.items() if length == 0]
    if empty_items:
        print(f"[WARNING] 警告：评估前发现空列表，后续 np.stack 可能报错。空项: {', '.join(empty_items)}")

    au_pros, _ = calculate_au_pro(gts, predictions)
    pixel_rocauc = roc_auc_score(np.stack(pixel_labels), np.stack(pixel_preds))
    image_rocauc = roc_auc_score(np.stack(image_labels), np.stack(image_preds))

    img_f1 = calculate_f1_max(np.stack(image_labels), np.stack(image_preds))
    pix_f1 = calculate_f1_max(np.stack(pixel_labels), np.stack(pixel_preds))

    print(f"\n鉁?V2.1 {args.class_name} 娴嬬畻瀹屾垚锛乗nI-AUROC | I-F1 | P-AUROC | P-F1 | AUPRO@30%\n  {image_rocauc:.3f}  | {img_f1:.3f} |  {pixel_rocauc:.3f}  | {pix_f1:.3f} |   {au_pros[0]:.3f}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default='./datasets/mvtec2d', type=str)
    parser.add_argument('--class_name', default="carpet", type=str)
    parser.add_argument('--checkpoint_folder', default='./checkpoints/checkpoints_CFM_mvtec', type=str)
    parser.add_argument('--epochs_no', default=50, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--seed', default=3407, type=int)
    parser.add_argument(
        '--postproc_variant',
        default='A',
        choices=['A', 'B', 'C', 'D'],
        type=str,
        help='A: border suppression + gaussian_filter(sigma=2), B: no border suppression, C: no gaussian smoothing, D: neither',
    )
    parser.add_argument('--heatmap_dir', default='./outputs/anomaly_heatmaps', type=str)
    parser.add_argument('--max_heatmaps', default=6, type=int)
    args = parser.parse_args()
    infer_CFM(args)

