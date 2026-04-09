export CUDA_VISIBLE_DEVICES=0

epochs=50
batch_size=4

# 🌟 算力压榨计划：一次性串行训练 5 个最难的纹理类别！
class_names=("carpet" "leather" "tile" "wood" "grid")

for class_name in "${class_names[@]}"
    do
        echo "================================================="
        echo "🔥 开始特训: $class_name (Epochs: $epochs)"
        echo "================================================="
        python cfm_training.py --class_name $class_name --epochs_no $epochs --batch_size $batch_size 
    done