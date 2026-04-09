import argparse
import os
import torch
import wandb
import numpy as np
from itertools import chain
from tqdm import tqdm, trange

os.environ["WANDB_MODE"] = "offline"

from models.features import MultimodalFeatures
from models.dataset import get_data_loader
from models.feature_transfer_nets import DynamicPromptSIREN, MultiScale_SpatialDecoder

def set_seeds(sid):
    np.random.seed(sid)
    torch.manual_seed(sid)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(sid)
        torch.cuda.manual_seed_all(sid)

def train_CFM(args):
    set_seeds(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_name = f'V2_Unfrozen_{args.class_name}_{args.epochs_no}ep'
    wandb.init(project='crossmodal-feature-mappings', name=model_name, reinit=True)
    
    train_loader = get_data_loader("train", class_name=args.class_name, img_size=224, dataset_path=args.dataset_path, batch_size=args.batch_size, shuffle=True)
    feature_extractor = MultimodalFeatures()

    CFM_RGBtoFreq = DynamicPromptSIREN(in_features=448, out_features=1152).to(device)
    CFM_FreqtoRGB = MultiScale_SpatialDecoder(in_features=1152, out_features=448).to(device)
    
    for param in feature_extractor.deep_feature_extractor.freq_backbone.parameters():
        param.requires_grad = True

    # ==========================================
    # 🌟 V2.2 核心优化器：分层调教策略
    # ==========================================
    # 提取解冻的 ResNet 参数
    backbone_params = [p for n, p in feature_extractor.rgb_backbone.named_parameters() if p.requires_grad]
    # 提取其他的网络参数
    other_params = chain(
        CFM_RGBtoFreq.parameters(), 
        CFM_FreqtoRGB.parameters(), 
        feature_extractor.deep_feature_extractor.freq_backbone.parameters()
    )

    # 区别对待学习率！
    optimizer = torch.optim.Adam([
        {'params': backbone_params, 'lr': 1e-5}, # Backbone 微调：防止毁掉基础特征
        {'params': other_params, 'lr': args.lr}   # 正常 1e-3 训练
    ])
    
    metric = torch.nn.CosineSimilarity(dim=-1, eps=1e-06)
    print(f"\n🔥 启动 V2.2 硬刚版: ResNet(Layer3解冻) + 分层学习率 | 类别: {args.class_name}")

    for epoch in trange(args.epochs_no, desc=f'Training {args.class_name}'):
        # 确保每轮骨干网络都在 train 模式
        feature_extractor.rgb_backbone.train() 
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}', leave=False)
        for (rgb, freq_img), _ in pbar:
            rgb, freq_img = rgb.to(device), freq_img.to(device)
            
            CFM_RGBtoFreq.train()
            CFM_FreqtoRGB.train()
            feature_extractor.deep_feature_extractor.freq_backbone.train() 

            if args.batch_size == 1:
                rgb_patch, freq_patch = feature_extractor.get_features_maps(rgb, freq_img)
            else:
                rgb_patches, freq_patches = [], []
                for i in range(rgb.shape[0]):
                    rp, fp = feature_extractor.get_features_maps(rgb[i].unsqueeze(dim=0), freq_img[i].unsqueeze(dim=0))
                    rgb_patches.append(rp)
                    freq_patches.append(fp)
                rgb_patch = torch.stack(rgb_patches, dim=0)
                freq_patch = torch.stack(freq_patches, dim=0)
            
            pred_freq = CFM_RGBtoFreq(rgb_patch) 
            pred_rgb = CFM_FreqtoRGB(freq_patch) 
            
            gamma = 2.0 
            base_loss_rgb = 1 - metric(pred_rgb, rgb_patch)     
            base_loss_freq = 1 - metric(pred_freq, freq_patch)  
            
            focal_weight_rgb = torch.pow(base_loss_rgb.detach(), gamma)
            focal_weight_freq = torch.pow(base_loss_freq.detach(), gamma)
            
            loss_FreqtoRGB = (focal_weight_rgb * base_loss_rgb).mean()
            loss_RGBtoFreq = (focal_weight_freq * base_loss_freq).mean()
            total_loss = loss_FreqtoRGB + loss_RGBtoFreq
            
            if not torch.isnan(total_loss) and not torch.isinf(total_loss):
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
            pbar.set_postfix({'Loss': f"{total_loss.item():.5f}"})

    directory = f'{args.checkpoint_savepath}/{args.class_name}'
    if not os.path.exists(directory): os.makedirs(directory)
    
    torch.save(CFM_RGBtoFreq.state_dict(), os.path.join(directory, f'CFM_RGBtoFreq_{model_name}.pth'))
    torch.save(CFM_FreqtoRGB.state_dict(), os.path.join(directory, f'CFM_FreqtoRGB_{model_name}.pth'))
    torch.save(feature_extractor.deep_feature_extractor.freq_backbone.state_dict(), os.path.join(directory, f'FreqBackbone_{model_name}.pth'))
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--dataset_path', default='./datasets/AITEX', type=str)
    parser.add_argument('--checkpoint_savepath', default='./checkpoints/checkpoints_CFM_mvtec', type=str)
    parser.add_argument('--class_name', default="aitex", type=str)
    parser.add_argument('--epochs_no', default=50, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--seed', default=3407, type=int)
    args = parser.parse_args()
    train_CFM(args)