import argparse
import os
import random

import cv2
import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, roc_auc_score
from tqdm import tqdm

from models.seg_dataset import get_seg_data_loader
from models.seg_models import CrossModalDBGSeg
from utils.metrics_utils import calculate_au_pro


def set_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_f1_max(labels, scores):
    precision, recall, _ = precision_recall_curve(labels, scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    return float(f1_scores[np.argmax(f1_scores)])


def compute_topk_image_score(prob_map, top_ratio=0.01):
    flat_scores = prob_map.reshape(-1)
    top_k = max(1, int(np.ceil(flat_scores.size * top_ratio)))
    top_values = np.partition(flat_scores, -top_k)[-top_k:]
    return float(top_values.mean())


def save_heatmap(prob_map, rgb_path, output_dir, class_name, split_name, save_index):
    sample_name = os.path.splitext(os.path.basename(rgb_path))[0]
    source_folder = os.path.basename(os.path.dirname(rgb_path))

    normalized_map = prob_map - prob_map.min()
    if normalized_map.max() > 0:
        normalized_map = normalized_map / normalized_map.max()

    heatmap_uint8 = np.uint8(normalized_map * 255.0)
    heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    save_folder = os.path.join(output_dir, class_name, split_name)
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, f"{save_index:03d}_{source_folder}_{sample_name}.png")
    cv2.imwrite(save_path, heatmap_bgr)


def evaluate_segmentation(args):
    set_deterministic(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_loader = get_seg_data_loader(
        split=args.eval_split,
        class_name=args.class_name,
        dataset_path=args.dataset_path,
        img_size=224,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
    )

    checkpoint_path = os.path.join(
        args.checkpoint_folder,
        args.class_name,
        f"CM_DBG_Seg_{args.class_name}_{args.epochs_no}ep_best.pth",
    )

    model = CrossModalDBGSeg(image_size=224).to(device)
    model.feature_extractor.device = device

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    gts = []
    predictions = []
    image_labels = []
    image_preds = []
    pixel_labels = []
    pixel_preds = []
    saved_heatmaps = 0

    with torch.no_grad():
        for (rgb, freq_img), gt_mask, label, rgb_path in tqdm(
            data_loader,
            desc=f"Evaluating {args.class_name} ({args.eval_split})",
            leave=False,
        ):
            rgb = rgb.to(device)
            freq_img = freq_img.to(device)
            gt_mask = gt_mask.to(device)

            logits = model(rgb, freq_img)
            probs = torch.sigmoid(logits)

            batch_size = rgb.shape[0]
            for sample_idx in range(batch_size):
                prob_map = probs[sample_idx, 0].detach().cpu().numpy()
                gt_map = gt_mask[sample_idx, 0].detach().cpu().numpy()
                sample_label = int(label[sample_idx].item())
                sample_path = rgb_path[sample_idx]

                gts.append(gt_map)
                predictions.append(prob_map)
                image_labels.append(sample_label)
                image_preds.append(compute_topk_image_score(prob_map, top_ratio=0.01))
                pixel_labels.extend(gt_map.reshape(-1))
                pixel_preds.extend(prob_map.reshape(-1))

                if args.heatmap_dir and args.max_heatmaps > 0:
                    if sample_label == 1 and saved_heatmaps < args.max_heatmaps:
                        save_heatmap(
                            prob_map,
                            sample_path,
                            args.heatmap_dir,
                            args.class_name,
                            args.eval_split,
                            saved_heatmaps,
                        )
                        saved_heatmaps += 1

    image_labels_np = np.asarray(image_labels)
    image_preds_np = np.asarray(image_preds)
    pixel_labels_np = np.asarray(pixel_labels)
    pixel_preds_np = np.asarray(pixel_preds)

    has_image_anomalies = np.unique(image_labels_np).size > 1
    has_pixel_anomalies = np.unique(pixel_labels_np).size > 1

    if has_image_anomalies:
        image_rocauc = roc_auc_score(image_labels_np, image_preds_np)
        image_f1 = calculate_f1_max(image_labels_np, image_preds_np)
    else:
        image_rocauc = float("nan")
        image_f1 = float("nan")

    if has_pixel_anomalies:
        pixel_rocauc = roc_auc_score(pixel_labels_np, pixel_preds_np)
        pixel_f1 = calculate_f1_max(pixel_labels_np, pixel_preds_np)
        au_pro_30, _ = calculate_au_pro(gts, predictions, integration_limit=[0.3])
        au_pro_30 = float(au_pro_30[0])
    else:
        pixel_rocauc = float("nan")
        pixel_f1 = float("nan")
        au_pro_30 = float("nan")

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Split: {args.eval_split}")
    print("I-AUROC | I-F1 | P-AUROC | P-F1 | AUPRO@30%")
    print(
        f"{image_rocauc:.3f} | {image_f1:.3f} | "
        f"{pixel_rocauc:.3f} | {pixel_f1:.3f} | {au_pro_30:.3f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CM-DBG-Seg on AITEX")
    parser.add_argument("--dataset_path", required=True, type=str)
    parser.add_argument("--class_name", required=True, type=str)
    parser.add_argument("--checkpoint_folder", required=True, type=str)
    parser.add_argument("--epochs_no", default=50, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--seed", default=3407, type=int)
    parser.add_argument("--eval_split", required=True, choices=["validation", "test"], type=str)
    parser.add_argument("--heatmap_dir", default="", type=str)
    parser.add_argument("--max_heatmaps", default=0, type=int)
    evaluate_segmentation(parser.parse_args())
