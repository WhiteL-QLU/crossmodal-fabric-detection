import glob
from PIL import Image
import numpy as np
from tqdm import tqdm

mask_files = glob.glob('./datasets/ITD/ITD/*/ground_truth/*/*.png')

dead_masks = 0  # 全黑掩码计数
alive_masks = 0 # 正常掩码计数
alive_values = set() # 记录正常的掩码是用什么数字标注的 (1 还是 255)

print(f"\n🔍 架构师全量排雷启动 | 共捕获掩码文件: {len(mask_files)} 个")

for m in tqdm(mask_files, desc="扫描掩码矩阵"):
    img = np.array(Image.open(m).convert('L'))
    uniques = np.unique(img)
    
    if len(uniques) == 1 and uniques[0] == 0:
        dead_masks += 1
    else:
        alive_masks += 1
        alive_values.add(uniques[-1]) # 提取异常区域的像素值

print("\n" + "="*40)
print(f"📊 排雷报告出炉:")
print(f"💀 彻底失效的纯黑掩码 (Dead): {dead_masks} 张")
print(f"🟢 包含真实标注的掩码 (Alive): {alive_masks} 张")
if alive_masks > 0:
    print(f"🏷️ 真实的异常标签像素值: {alive_values}")
print("="*40 + "\n")