export CUDA_VISIBLE_DEVICES=0

python3 main_bp_collaboration.py \
    --batch_size 64 \
    --data <your_path>/imagenet \
    --data_corruption <your_path>/imagenet-c \
    --data_v2 <your_path>/imageNet-v2 \
    --data_sketch <your_path>/imagenet-sketch \
    --data_rendition <your_path>/imagenet-r \
    --output ./outputs/bp_collaboration \
    --resume weights/original.pth \
    --algorithm 'eta-cola' \
    --tag '_test'

# testV4_3agents_showEfficiency_onGauss_pzlEnv_oldNorm
# _testSave_testCounterPart_3agents_adaptWholeType_lr0.001_iidByNonShuffle_pzlEnv_oldNorm
# _testSave_3agents_adaptWholeType_iidByNonShuffle_wd0.2_pzlEnv_oldNorm
# _test_3agents_iidByNonShuffle_wd0.0_pzlEnv_oldNorm