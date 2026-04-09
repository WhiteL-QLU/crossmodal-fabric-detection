import os
import time

# 这是 MVTec AD 中除了 carpet 之外，最考验纹理特征和频域去噪能力的 4 大类别
texture_classes = ['grid', 'leather', 'wood', 'tile']

epochs = 50
batch_size = 4
seed = 3407
dataset_path = "./datasets/mvtec2d"

print("🚀 [架构师防漏检测] 开启 MVTec 纹理阵线盲测...")
time.sleep(2)

for cls in texture_classes:
    print(f"\n{'='*50}")
    print(f"🔥 正在全自动激战类别: 【{cls.upper()}】")
    print(f"{'='*50}\n")
    
    # 1. 自动启动训练 (Training)
    print(f"➡️ [Step 1] 开始训练 {cls} 的专属权重...")
    train_cmd = f"python cfm_training.py --class_name {cls} --epochs_no {epochs} --batch_size {batch_size} --seed {seed} --dataset_path {dataset_path}"
    os.system(train_cmd)
    
    # 2. 自动启动推理 (Inference) - 使用我们今晚确立的 0.998 终极逻辑
    print(f"\n➡️ [Step 2] 开始使用终极融合版进行 {cls} 推理算分...")
    infer_cmd = f"python cfm_inference.py --class_name {cls} --epochs_no {epochs} --batch_size {batch_size} --seed {seed} --dataset_path {dataset_path}"
    os.system(infer_cmd)

print("\n🎉 架构师汇报：四大纹理类别全部跑测完毕！请前往 results 文件夹检阅战果！")