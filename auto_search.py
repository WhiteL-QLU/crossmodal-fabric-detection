import os
import time

# 🌟 终极免打扰模式：强行按死 wandb 离线状态
os.environ["WANDB_MODE"] = "offline"

# 3 种学习率 × 3 个玄学种子 = 9 种组合
learning_rates = [0.001, 0.0005, 0.0001]
seeds = [115, 42, 3407]

# 统一参数设定（建议先用 50 轮，2D 架构收敛极快）
EPOCHS = 50
BATCH_SIZE = 4

# 初始化成绩单文件
result_file = "final_results.txt"
os.system(f"echo '=== 织物缺陷检测 (Carpet) 全自动网格搜索成绩单 ===' > {result_file}")

print("🚀 赛博替身已启动（已免疫 Wandb 拦截弹窗）！\n")
print("⚠️ 预警：此脚本旨在快速寻找最佳参数组合，模型权重将以同名覆盖，请根据最终成绩单重新定点训练最佳模型！\n")

total_tasks = len(learning_rates) * len(seeds)
current_task = 1

for lr in learning_rates:
    for seed in seeds:
        print(f"🔄 [任务 {current_task}/{total_tasks}] 正在训练: 学习率 LR={lr}, 种子 Seed={seed}")
        
        # 1. 自动执行训练指令
        train_cmd = f"python cfm_training.py --class_name carpet --epochs_no {EPOCHS} --batch_size {BATCH_SIZE} --dataset_path ./datasets/mvtec2d --lr {lr} --seed {seed}"
        os.system(train_cmd)
        
        print(f"✅ [任务 {current_task}/{total_tasks}] 训练完成，正在拉去考场自动打分...")
        
        # 2. 把当前参数组合的标题写入成绩单
        os.system(f"echo '\n\n--------------------------------------------' >> {result_file}")
        os.system(f"echo '🧪 参数组合: LR={lr}, Seed={seed}' >> {result_file}")
        
        # 3. 自动执行推理指令 (🌟 修复：严丝合缝对齐 seed)
        infer_cmd = f"python cfm_inference.py --class_name carpet --epochs_no {EPOCHS} --batch_size {BATCH_SIZE} --dataset_path ./datasets/mvtec2d --seed {seed} >> {result_file}"
        os.system(infer_cmd)
        
        print(f"📝 成绩已登记进 {result_file}，准备下一轮...\n")
        time.sleep(2) 
        current_task += 1

print("🎉 报告大姐！所有 9 种组合全部跑完并已记录！请查阅 final_results.txt。")