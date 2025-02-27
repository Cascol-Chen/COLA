export CUDA_VISIBLE_DEVICES=0

python3 main_fp_agents.py \
    --batch_size 64 \
    --data <your_path>/imagenet \
    --data_corruption <your_path>/imagenet-c \
    --data_v2 <your_path>imageNet-v2 \
    --data_sketch <your_path>/imagenet-sketch \
    --data_rendition <your_path>/imagenet-r \
    --data_adv <your_path>/imagenet-a \
    --output ./outputs/fp_collaboration \
    --resume <saved_vectors_path>/model \
    --exp_type 'continual' \
    --algorithm 'cola-fp' \
    --tag '_test'

# _test_ImageNetCOriginal_with_original_temperatureD5_v2_1round_pzlEnv_oldNorm
# _test_imagenetC_with_original_temperatureD10_v2_1round_pzlEnv_oldNorm