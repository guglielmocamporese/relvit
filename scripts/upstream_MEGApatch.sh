#!/bin/bash

# Retrieve dataset
MODE="train"
BACKBONE="vit"
MODEL_CKPT=""
PATCH_SIZE=-1
BATCH_SIZE=256
METRIC_MONITOR="val_acc_rel"
MODEL_SIZE="small"
EXP_ID=""
OPTIMIZER="adamw"
LR="5e-4"
EPOCHS=100
NUM_GPUS=1
PROJ_DROPOUT=0
DROP_PATH_RATE=0.0
DROPOUT=0.1
WEIGHT_DECAY=0.0
NUM_WORKERS="8"
DATA_BASE_PATH="./datasets/data" 
PATCH_TRANS="colJitter:0.8-grayScale:0.2"
SIDE_MEGAPATCHES=5
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        --dataset)
        DATASET="$2"
        shift # past argument
        shift # past value
        ;;
        --dropout)
        DROPOUT="$2"
        shift # past argument
        shift # past value
        ;;
        --weight_decay)
        WEIGHT_DECAY="$2"
        shift # past argument
        shift # past value
        ;;
        --optimizer)
        OPTIMIZER="$2"
        shift # past argument
        shift # past value
        ;;
        --lr)
        LR="$2"
        shift # past argument
        shift # past value
        ;;
        --drop_path_rate)
        DROP_PATH_RATE="$2"
        shift # past argument
        shift # past value
        ;;
        --mode)
        MODE="$2"
        shift # past argument
        shift # past value
        ;;
        --backbone)
        BACKBONE="$2"
        shift # past argument
        shift # past value
        ;;
        --model_size)
        MODEL_SIZE="$2"
        shift # past argument
        shift # past value
        ;;
        --model_checkpoint)
        MODEL_CKPT="$2"
        shift # past argument
        shift # past value
        ;;
        --epochs)
        EPOCHS="$2"
        shift # past argument
        shift # past value
        ;;
        --num_gpus)
		NUM_GPUS="$2"
		shift # past argument
		shift # past value
		;;
        --patch_size)
        PATCH_SIZE="$2"
        shift # past argument
        shift # past value
        ;;
        --batch_size)
        BATCH_SIZE="$2"
        shift # past argument
        shift # past value
        ;;
        --metric_monitor)
        METRIC_MONITOR="$2"
        shift # past argument
        shift # past value
        ;;
        --exp_id)
        EXP_ID="$2"
        shift # past argument
        shift # past value
        ;;
        --proj_dropout)
        PROJ_DROPOUT="$2"
        shift # past argument
        shift # past value
        ;;
        --num_workers)
		NUM_WORKERS="$2"
		shift # past argument
		shift # past value
		;;
		--data_base_path)
		DATA_BASE_PATH="$2"
		shift # past argument
		shift # past value
		;;
        --patch_trans)
        PATCH_TRANS="$2"
        shift
        shift
        ;;
        --side_megapatches)
        SIDE_MEGAPATCHES="$2"
        shift
        shift
        ;;
    *)
		echo "$1 argument not allowed"
		exit 1
    esac

done
set -- "${POSITIONAL[@]}" # restore positional parameters

# Run main
python main.py \
    --seed 35771 \
    --logger "wandb" \
    --mode "${MODE}" \
    --task "upstream" \
    --grad_clip_val 1.0 \
    --exp_id "${EXP_ID}" \
    --backbone "${BACKBONE}" \
    --model_size "${MODEL_SIZE}" \
    --num_gpus ${NUM_GPUS} \
    --epochs ${EPOCHS} \
    --dataset "${DATASET}" \
    --data_base_path ${DATA_BASE_PATH} \
    --batch_size ${BATCH_SIZE} \
    --num_workers "${NUM_WORKERS}"\
    --weight_decay ${WEIGHT_DECAY} \
    --optimizer "${OPTIMIZER}" \
    --lr "${LR}" \
    --drop_path_rate ${DROP_PATH_RATE} \
    --proj_dropout ${PROJ_DROPOUT} \
    --metric_monitor "${METRIC_MONITOR}" \
    --model_checkpoint "${MODEL_CKPT}" \
    --patch_size "${PATCH_SIZE}" \
    --dropout ${DROPOUT} \
    --attn_mask "" \
    --use_relations \
    --use_abs_positions \
    --patch_trans "${PATCH_TRANS}" \
    --mega_patches \
    --side_megapatches ${SIDE_MEGAPATCHES} \

