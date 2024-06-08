#!/bin/bash
export TORCH_USE_CUDA_DSA=1
# abort entire script on error
set -e

DATA_ROOT=/home/wangxiaoqi/EK100_320p_15sec_30fps_libx264
AVION_ROOT=/home/wangxiaoqi/SMS
EXP_PATH=$AVION_ROOT/experiments/vitbl_no_thres
if [ ! -d $EXP_PATH ]; then
    mkdir $EXP_PATH
fi

#cp this shell file to experiment folder
SCRIPT_PATH="$0"
SCRIPT_NAME=$(basename "$SCRIPT_PATH")
cp "$SCRIPT_PATH" "$EXP_PATH/$SCRIPT_NAME"
# python scripts/main_lavila_finetune_mir.py\
#     --root $DATA_ROOT/ \
#     --train-metadata $DATA_ROOT/epic-kitchens-100-annotations/retrieval_annotations/EPIC_100_retrieval_train.csv \
#     --val-metadata $DATA_ROOT/epic-kitchens-100-annotations/retrieval_annotations/EPIC_100_retrieval_test.csv \
#     --relevancy-path $DATA_ROOT/epic-kitchens-100-annotations/retrieval_annotations/relevancy/caption_relevancy_EPIC_100_retrieval_test.pkl \
#     --output-dir ./ \
#     --video-chunk-length 15 --use-flash-attn \
#     --grad-checkpointing \
#     --use-fast-conv1 \
#     --batch-size 64 \
#     --no-gather-with-grad \
#     --fused-decode-crop \
#     --pretrain-model $AVION_ROOT/experiments/AVION_pretrain_lavila_vitb_best.pt \
#     --output-dir $EXP_PATH 2>&1 | tee $EXP_PATH/log.txt

PYTHONPATH=./third_party/decord/python/ NCCL_P2P_LEVEL=NVL torchrun --nproc_per_node=4 scripts/ammplus_finetune.py\
    --root $DATA_ROOT/ \
    --train-metadata $DATA_ROOT/epic-kitchens-100-annotations/retrieval_annotations/EPIC_100_retrieval_train.csv \
    --val-metadata $DATA_ROOT/epic-kitchens-100-annotations/retrieval_annotations/EPIC_100_retrieval_test.csv \
    --relevancy-train $DATA_ROOT/epic-kitchens-100-annotations/retrieval_annotations/relevancy/caption_relevancy_EPIC_100_retrieval_train.pkl \
    --relevancy-test $DATA_ROOT/epic-kitchens-100-annotations/retrieval_annotations/relevancy/caption_relevancy_EPIC_100_retrieval_test.pkl \
    --output-dir ./ \
    --video-chunk-length 15 --use-flash-attn \
    --grad-checkpointing \
    --use-fast-conv1 \
    --model CLIP_VITL14 \
    --batch-size 60 \
    --epoch 100 \
    --loss-margin 0.6 \
    --loss-thres 0.1 \
    --lr 2e-5 \
    --lr-start 1e-6 \
    --lr-end 1e-6 \
    --seed 3407 \
    --fused-decode-crop \
    --dist-backend 'nccl' \
    --use-multi-epochs-loader \
    --pretrain-model $AVION_ROOT/experiments/avion_pretrain_lavila_vitl_best.pt \
    --output-dir $EXP_PATH 2>&1 | tee $EXP_PATH/log.txt