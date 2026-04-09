import argparse
import os
import torch
import numpy as np
import cv2
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

from models.features import MultimodalFeatures
from models.dataset import get_data_loader
from models.feature_transfer_nets import DynamicPromptSIREN

def set_seeds(sid=42):
    np.random.seed(sid)
    torch.manual_seed(sid)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(sid)
        torch.cuda.manual_seed_all(sid)

def denormalize(img):
    """恢复 RGB 图片显示"""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img.transpose(1, 2, 0) * std + mean) * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)

def visualize_CFM(args):
    set_seeds(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_loader = get_data_loader("test", class_name=args.class_name, img_size=224, dataset_path=args.dataset_path, batch_size=1)
    feature_extractor = MultimodalFeatures()

    CFM_RGBtoFreq = DynamicPromptSIREN(in_features=768, out_features=1152).to(device)
    CFM_FreqtoRGB = DynamicPromptSIREN(in_features=1152, out_features=768).to(device)

    model_name = f'{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs'
    CFM_RGBtoFreq.load_state_dict(torch.load(rf'{args.checkpoint_folder}/{args.class_name}/CFM_RGBtoFreq_{model_name}.pth'))
    CFM_FreqtoRGB.load_state_dict(torch.load(rf'{args.checkpoint_folder}/{args.class_name}/CFM_FreqtoRGB_{model_name}.pth'))
    
    try:
        feature_extractor.deep_feature_extractor.freq_backbone.load_state_dict(torch.load(rf'{args.checkpoint_folder}/{args.class_name}/FreqBackbone_{model_name}.pth'))
    except: pass

    CFM_RGBtoFreq.eval()
    CFM_FreqtoRGB.eval()
    feature_extractor.deep_feature_extractor.freq_backbone.eval()
    
    save_dir = os.path.join(args.vis_folder, args.class_name)
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    cos_sim = torch.nn.CosineSimilarity(dim=1)

    # 跑前 20 张图，包含正常和异常
    for i, ((rgb, freq_img), gt, label, rgb_path) in enumerate(tqdm(test_loader, desc='Visualizing')):
        if i >= 20: break 
        
        rgb_gpu, freq_gpu = rgb.to(device), freq_img.to(device)

        with torch.no_grad():
            rgb_patch, freq_patch = feature_extractor.get_features_maps(rgb_gpu, freq_gpu)
            pred_freq = CFM_RGBtoFreq(rgb_patch)
            pred_rgb = CFM_FreqtoRGB(freq_patch)

            score_freq = 1 - cos_sim(pred_freq, freq_patch).reshape(28, 28)
            score_rgb = 1 - cos_sim(pred_rgb, rgb_patch).reshape(28, 28)
            
            # 融合与后处理
            cos_comb = (score_rgb * score_freq).reshape(1, 1, 28, 28)
            cos_comb = torch.nn.functional.interpolate(cos_comb, size=(224, 224), mode='bilinear', align_corners=False)
            anomaly_map = cos_comb.squeeze().cpu().numpy()
            anomaly_map = gaussian_filter(anomaly_map, sigma=4) # 稍微加大平滑，让图更漂亮

            # 绘图
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            
            # 1. 原图
            ax[0].imshow(denormalize(rgb.squeeze().numpy()))
            ax[0].set_title(f'Original ({"Ano" if label.item() else "Norm"})')
            ax[0].axis('off')
            
            # 2. GT
            ax[1].imshow(gt.squeeze().numpy(), cmap='gray')
            ax[1].set_title('Ground Truth')
            ax[1].axis('off')
            
            # 3. OpenCV 伪彩色热力图
            norm_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
            heatmap = cv2.applyColorMap((norm_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            ax[2].imshow(heatmap)
            ax[2].set_title('Anomaly Heatmap')
            ax[2].axis('off')
            
            file_name = os.path.basename(rgb_path[0]).split('.')[0]
            plt.savefig(os.path.join(save_dir, f'{file_name}_vis.jpg'), bbox_inches='tight')
            plt.close()

    print(f"\n✅ 可视化完成！路径: {save_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default='./datasets/mvtec2d', type=str)
    parser.add_argument('--class_name', default="carpet", type=str)
    parser.add_argument('--checkpoint_folder', default='./checkpoints/checkpoints_CFM_mvtec', type=str)
    parser.add_argument('--vis_folder', default='./results/visualizations_mvtec', type=str)
    parser.add_argument('--epochs_no', default=50, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--seed', default=3407, type=int)
    args = parser.parse_args()
    visualize_CFM(args)