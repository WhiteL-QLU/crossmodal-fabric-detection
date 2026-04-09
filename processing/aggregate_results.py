import os
import argparse
import numpy as np
import pandas as pd

metrics = ["AUPRO@30%", "AUPRO@10%", "AUPRO@5%", "AUPRO@1%", "P-AUROC", "I-AUROC"]

def read_md_files(directory):
    aggregated_results = []
    actual_classes = [] 
    
    # 获取所有 md 文件
    md_files = sorted([filename for filename in os.listdir(directory) if filename.endswith(".md")])

    for filename in md_files:
        # 🌟 核心修复：现在的名字是 carpet_50ep_4bs_Ablation.md
        # 所以类名在用 '_' 分割后的第一个位置 [0]
        class_name = filename.split('_')[0] 
        actual_classes.append(class_name)
        
        filepath = os.path.join(directory, filename)
        with open(filepath, "r", encoding="utf-8") as file:
            lines = file.read().strip().split('\n')
            file_contents = lines[-1].split('|') 
            file_contents_num = [float(res.strip()) for res in file_contents if res.strip()] 
            aggregated_results.append(file_contents_num)
            
    return np.array(aggregated_results), actual_classes

def produce_table(args):
    data, actual_classes = read_md_files(args.quantitative_folder)
    
    if len(data) == 0:
        print(f"⚠️ 在 {args.quantitative_folder} 中没有找到任何 .md 文件，请确认是否已经跑完推理！")
        return

    results = pd.DataFrame(data, index=actual_classes, columns=metrics)
    results.loc['Mean'] = results.mean(axis=0)

    print("\n" + "="*60)
    print("🏆 论文专用 LaTeX 表格代码")
    print("="*60)
    print(results.to_latex(float_format="%.3f"))
    
    print("\n" + "="*60)
    print("📊 方便肉眼查看的控制台表格")
    print("="*60)
    print(results.round(3).to_markdown())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a summary table with all results.')
    # 🌟 修复：路径统一改为从根目录出发的 ./results
    parser.add_argument('--quantitative_folder', default='./results/quantitatives_mvtec', type=str,
                        help='Path to the folder from which to fetch the quantitatives.')
    args = parser.parse_args()
    
    produce_table(args)