export CUDA_VISIBLE_DEVICES=0

epochs=50
batch_size=4

# 🌟 验收计划：一次性评估刚才跑完的 5 个类别！
class_names=("carpet" "leather" "tile" "wood" "grid")

for class_name in "${class_names[@]}"
    do
        echo "================================================="
        echo "🎯 开始验收打分: $class_name (Epochs: $epochs)"
        echo "================================================="
        python cfm_inference.py --class_name $class_name --epochs_no $epochs --batch_size $batch_size
    done