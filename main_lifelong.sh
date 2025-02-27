export CUDA_VISIBLE_DEVICES=0

python3 main_lifelong.py \
    --seed 2022 \
    --batch_size 64 \
    --data <your_path>/imagenet \
    --data_corruption <your_path>/imagenet-c \
    --data_v2 <your_path>/imageNet-v2 \
    --data_sketch <your_path>/imagenet-sketch \
    --data_rendition <your_path>/imagenet-r \
    --data_adv <your_path>/imagenet-a \
    --output ./outputs/lifelong \
    --resume weights/original.pth \
    --algorithm 'eta-cola' \
    --tag '_test'