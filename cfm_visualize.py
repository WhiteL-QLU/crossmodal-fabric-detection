import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from models.features import MultimodalFeatures
from models.dataset import get_data_loader
from models.feature_transfer_nets import DynamicPromptSIREN

def denormalize(tensor):
    # 反转 ImageNet 归一化，为了让原图在画板上正常显示颜色
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img

def run_visualization(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 强制 batch_size=1 以便逐张抓取异常图片
    test_loader = get_data_loader("test", class_name=args.class_name, img_size=224, dataset_path=args.dataset_path, batch_size=1)
    feature_extractor = MultimodalFeatures()

    CFM_RGBtoFreq = DynamicPromptSIREN(in_features=768, out_features=1152).to(device)
    CFM_FreqtoRGB = DynamicPromptSIREN(in_features=1152, out_features=768).to(device)

    model_name = f'{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs'
    CFM_RGBtoFreq.load_state_dict(torch.load(rf'{args.checkpoint_folder}/{args.class_name}/CFM_RGBtoFreq_{model_name}.pth'))
    CFM_FreqtoRGB.load_state_dict(torch.load(rf'{args.checkpoint_folder}/{args.class_name}/CFM_FreqtoRGB_{model_name}.pth'))

    CFM_RGBtoFreq.eval()
    CFM_FreqtoRGB.eval()
    
    cos_sim = torch.nn.CosineSimilarity(dim=1)
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    
    row = 0
    print("\n🔍 正在扫描测试集，抓取异常样本进行可视化渲染...")
    for (rgb, freq_img), gt, label, rgb_path in test_loader:
        if label.item() == 0:
            continue # 跳过正常布料，我们只看模型是怎么抓瑕疵的
            
        rgb, freq_img = rgb.to(device), freq_img.to(device)
        
        with torch.no_grad():
            rgb_patch, freq_patch = feature_extractor.get_features_maps(rgb, freq_img)
            pred_rgb = CFM_FreqtoRGB(freq_patch)
            
            # 【使用终极解耦逻辑：纯空间流定位】
            score_rgb = 1 - cos_sim(pred_rgb, rgb_patch).reshape(1, 1, 28, 28)
            cos_spatial = torch.nn.functional.interpolate(score_rgb, size=(224, 224), mode='bilinear', align_corners=False).squeeze().cpu().numpy()
            
            margin = int(224 * 0.05)
            cos_spatial[:margin, :] = 0
            cos_spatial[-margin:, :] = 0
            cos_spatial[:, :margin] = 0
            cos_spatial[:, -margin:] = 0
            
            cos_spatial = gaussian_filter(cos_spatial, sigma=2)
            
            # 仅用于可视化的归一化，让热力图颜色对比更强烈
            heatmap = (cos_spatial - cos_spatial.min()) / (cos_spatial.max() - cos_spatial.min() + 1e-8)
            
            orig_img = denormalize(rgb.squeeze())
            gt_img = gt.squeeze().cpu().numpy()
            
            # 绘图逻辑
            axes[row, 0].imshow(orig_img)
            axes[row, 0].set_title(f"Original: {os.path.basename(rgb_path[0])}")
            axes[row, 0].axis('off')
            
            axes[row, 1].imshow(gt_img, cmap='gray')
            axes[row, 1].set_title("Ground Truth (1-pixel width)")
            axes[row, 1].axis('off')
            
            im = axes[row, 2].imshow(heatmap, cmap='jet')
            axes[row, 2].set_title("Predicted Heatmap")
            axes[row, 2].axis('off')
            
        row += 1
        if row >= 3:
            break
            
    save_path = "aitex_visualization_results.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 渲染完毕！可视化对比图已保存至: {os.path.abspath(save_path)}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default='./datasets/AITEX', type=str)
    parser.add_argument('--class_name', default="aitex", type=str)
    parser.add_argument('--checkpoint_folder', default='./checkpoints/checkpoints_CFM_mvtec', type=str)
    parser.add_argument('--epochs_no', default=50, type=int)
    # 强制继承之前训练时的 batch_size = 4 的模型路径参数，但内部强转 batch_size=1 进行绘图
    parser.add_argument('--batch_size', default=4, type=int) 
    args = parser.parse_args()
    run_visualization(args)