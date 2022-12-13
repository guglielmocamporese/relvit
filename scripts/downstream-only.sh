#!/bin/bash

# Retrieve dataset
MODE="train"
MODEL_CKPT=""
BACKBONE="vit"
PATCH_SIZE=-1
BATCH_SIZE=256
EPOCHS=100
NUM_GPUS=1
EXP_ID=""
LR="5e-4"
OPTIMIZER="adamw"
MODEL_SIZE="small"
DROPOUT_PATH_RATE="0.0"
NUM_WORKERS="8"
PROJ_DROPOUT=0.0
DROPOUT=0.1
WEIGHT_DECAY=0.0
DATA_BASE_PATH="./datasets/data" 
PATCH_TRANS="None"
LABELS_PATH="" 
SEED=35771
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        --dataset)
		DATASET="$2"
		shift # past argument
		shift # past value
		;;
        --mode)
		MODE="$2"
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
        --lr)
		LR="$2"
		shift # past argument
		shift # past value
		;;
        --optimizer)
		OPTIMIZER="$2"
		shift # past argument
		shift # past value
		;;
        --model_checkpoint)
		MODEL_CKPT="$2"
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
        --exp_id)
		EXP_ID="$2"
		shift # past argument
		shift # past value
		;;
        --drop_path_rate)
		DROPOUT_PATH_RATE="$2"
		shift # past argument
		shift # past value
		;;
        --num_workers)
		NUM_WORKERS="$2"
		shift # past argument
		shift # past value
		;;
        --proj_dropout)
		PROJ_DROPOUT="$2"
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
        --labels_path)                                                                                                 
        LABELS_PATH="$2"                                                                                               
        shift                                                                                                          
        shift                                                                                                          
        ;;
        --seed)                                                                                                 
        SEED="$2"                                                                                               
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
    --seed ${SEED} \
    --mode "${MODE}" \
    --task "downstream" \
    --grad_clip_val 1.0 \
    --exp_id "${EXP_ID}" \
    --backbone "${BACKBONE}" \
    --num_gpus ${NUM_GPUS} \
    --epochs ${EPOCHS} \
    --dataset "${DATASET}" \
    --data_base_path ${DATA_BASE_PATH}\
    --batch_size ${BATCH_SIZE} \
    --num_workers "${NUM_WORKERS}" \
    --optimizer "${OPTIMIZER}" \
    --weight_decay ${WEIGHT_DECAY} \
    --lr "${LR}" \
    --proj_dropout ${PROJ_DROPOUT} \
    --drop_path_rate "${DROPOUT_PATH_RATE}" \
    --metric_monitor "val_acc_clf" \
    --model_checkpoint "${MODEL_CKPT}" \
    --model_size ${MODEL_SIZE} \
    --patch_size $PATCH_SIZE \
    --dropout ${DROPOUT} \
	--patch_trans "${PATCH_TRANS}" \
    --use_positional_embeddings \
    --use_supervision \
    --use_relations \
    --use_abs_positions \
	--labels_path "${LABELS_PATH}" \
	--use_clf_token 

