import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
from models.features import MultimodalFeatures
from models.dataset import get_data_loader
from models.feature_transfer_nets import DynamicPromptSIREN
from sklearn.metrics import roc_auc_score

def infer_CFM_Raw(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_loader = get_data_loader("test", class_name=args.class_name, img_size=224, dataset_path=args.dataset_path)
    feature_extractor = MultimodalFeatures()
    
    CFM_RGBtoFreq = DynamicPromptSIREN(in_features=768, out_features=1152).to(device)
    CFM_FreqtoRGB = DynamicPromptSIREN(in_features=1152, out_features=768).to(device)

    model_name = f'{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs'
    CFM_RGBtoFreq.load_state_dict(torch.load(rf'{args.checkpoint_folder}/{args.class_name}/CFM_RGBtoFreq_{model_name}.pth'))
    CFM_FreqtoRGB.load_state_dict(torch.load(rf'{args.checkpoint_folder}/{args.class_name}/CFM_FreqtoRGB_{model_name}.pth'))
    
    CFM_RGBtoFreq.eval()
    CFM_FreqtoRGB.eval()

    image_labels, image_preds = [], []
    cos_sim = torch.nn.CosineSimilarity(dim=1)

    for (rgb, freq_img), gt, label, rgb_path in tqdm(test_loader, desc=f'Evaluating {args.class_name} (RAW)', leave=False):
        rgb, freq_img = rgb.to(device), freq_img.to(device)

        with torch.no_grad():
            rgb_patch, freq_patch = feature_extractor.get_features_maps(rgb, freq_img)
            pred_freq = CFM_RGBtoFreq(rgb_patch)
            pred_rgb = CFM_FreqtoRGB(freq_patch)

            score_freq = 1 - cos_sim(pred_freq, freq_patch).reshape(28, 28)
            score_rgb = 1 - cos_sim(pred_rgb, rgb_patch).reshape(28, 28)
            
            # 【底层架构的改变】 纯净版：没有任何切除，没有高斯滤波，没有 Top-500
            cos_comb = torch.maximum(score_rgb, score_freq).reshape(1, 1, 28, 28)
            cos_comb = torch.nn.functional.interpolate(cos_comb, size=(224, 224), mode='bilinear', align_corners=False)
            cos_comb_np = cos_comb.squeeze().cpu().numpy()
            
            image_labels.append(label.cpu().numpy()) 
            # 图像级异常得分直接取最大值
            image_preds.append(cos_comb_np.max()) 

    image_rocauc = roc_auc_score(np.stack(image_labels), np.stack(image_preds))
    print(f"\n🔬 裸特征盲测完毕！\nRAW I-AUROC: {image_rocauc:.3f}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default='./datasets/AITEX', type=str)
    parser.add_argument('--class_name', default="aitex", type=str)
    parser.add_argument('--checkpoint_folder', default='./checkpoints/checkpoints_CFM_mvtec', type=str)
    parser.add_argument('--epochs_no', default=50, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    args = parser.parse_args()
    infer_CFM_Raw(args)