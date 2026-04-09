export CUDA_VISIBLE_DEVICES=0

epochs=50
batch_size=4

class_names=("cookie")

for class_name in "${class_names[@]}"
    do
        python cfm_training.py --class_name $class_name --epochs_no $epochs --batch_size $batch_size 
    done