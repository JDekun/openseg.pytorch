#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd $SCRIPTPATH
cd ../

DATA_DIR="../../input/openseg-cityscapes-gtfine"
SAVE_DIR="./result/cityscapes/checkpoints/"
BACKBONE="deepbase_resnet101_dilated8"

CONFIGS="configs/cityscapes/R_101_D_8.json"
CONFIGS_TEST="configs/cityscapes/R_101_D_8_TEST.json"
PRETRAINED_MODEL="../../input/pre-trained/resnet101-imagenet-openseg.pth"

MODEL_NAME="resnet_fcn_asp3_mep"
LOSS_TYPE="fs_auxce_loss_dc"
MEMORY_SIZE=8192
CHECKPOINTS_NAME="${MODEL_NAME}${MEMORY_SIZE}(183654)_${BACKBONE}_$(date +%F_%H-%M-%S)"
LOG_FILE="./experiment/log1/cityscapes/${CHECKPOINTS_NAME}.log"
echo "Logging to $LOG_FILE"
mkdir -p `dirname $LOG_FILE`
MAX_ITERS=40000

python -u main.py --configs ${CONFIGS} \
                      --drop_last y \
                      --phase train \
                      --gathered n \
                      --loss_balance y \
                      --log_to_file n \
                      --backbone ${BACKBONE} \
                      --model_name ${MODEL_NAME} \
                      --gpu 3 4 5 6  \
                      --train_batch_size 8\
                      --val_batch_size 4 \
                      --memory_size ${MEMORY_SIZE}\
                      --projector "layer_2" "layer_3" "layer_4"\
                      --data_dir ${DATA_DIR} \
                      --loss_type ${LOSS_TYPE} \
                      --max_iters ${MAX_ITERS} \
                      --checkpoints_name ${CHECKPOINTS_NAME} \
                      --pretrained ${PRETRAINED_MODEL} \
                      --distributed \
                      2>&1 | tee ${LOG_FILE}


MEMORY_SIZE=16384
CHECKPOINTS_NAME="${MODEL_NAME}${MEMORY_SIZE}(183654)_${BACKBONE}_$(date +%F_%H-%M-%S)"
LOG_FILE="./experiment/log1/cityscapes/${CHECKPOINTS_NAME}.log"
echo "Logging to $LOG_FILE"
mkdir -p `dirname $LOG_FILE`
MAX_ITERS=80000
BASE_LR=0.01
python -u main.py --configs ${CONFIGS} \
                      --drop_last y \
                      --phase train \
                      --gathered n \
                      --loss_balance y \
                      --log_to_file n \
                      --backbone ${BACKBONE} \
                      --model_name ${MODEL_NAME} \
                      --gpu 3 4 5 6  \
                      --train_batch_size 8\
                      --val_batch_size 4 \
                      --memory_size ${MEMORY_SIZE}\
                      --projector "layer_2" "layer_3" "layer_4"\
                      --data_dir ${DATA_DIR} \
                      --loss_type ${LOSS_TYPE} \
                      --max_iters ${MAX_ITERS} \
                      --checkpoints_name ${CHECKPOINTS_NAME} \
                      --pretrained ${PRETRAINED_MODEL} \
                      --distributed \
                      --base_lr ${BASE_LR} \
                      2>&1 | tee ${LOG_FILE}



MODEL_NAME="resnet_ocr_asp0_mep_in"
MEMORY_SIZE=16384
CHECKPOINTS_NAME="${MODEL_NAME}${MEMORY_SIZE}_${BACKBONE}_$(date +%F_%H-%M-%S)"
LOG_FILE="./experiment/log1/cityscapes/${CHECKPOINTS_NAME}.log"
echo "Logging to $LOG_FILE"
mkdir -p `dirname $LOG_FILE`
MAX_ITERS=40000
BASE_LR=0.01
python -u main.py --configs ${CONFIGS} \
                      --drop_last y \
                      --phase train \
                      --gathered n \
                      --loss_balance y \
                      --log_to_file n \
                      --backbone ${BACKBONE} \
                      --model_name ${MODEL_NAME} \
                      --gpu 3 4 5 6  \
                      --train_batch_size 8\
                      --val_batch_size 4 \
                      --memory_size ${MEMORY_SIZE}\
                      --projector "layer_1" \
                      --data_dir ${DATA_DIR} \
                      --loss_type ${LOSS_TYPE} \
                      --max_iters ${MAX_ITERS} \
                      --checkpoints_name ${CHECKPOINTS_NAME} \
                      --pretrained ${PRETRAINED_MODEL} \
                      --distributed \
                      --base_lr ${BASE_LR} \
                      2>&1 | tee ${LOG_FILE}


MODEL_NAME="resnet_fcn512_asp3_mep"
MEMORY_SIZE=16384
CHECKPOINTS_NAME="${MODEL_NAME}${MEMORY_SIZE}_${BACKBONE}_$(date +%F_%H-%M-%S)"
LOG_FILE="./experiment/log1/cityscapes/${CHECKPOINTS_NAME}.log"
echo "Logging to $LOG_FILE"
mkdir -p `dirname $LOG_FILE`
MAX_ITERS=40000
BASE_LR=0.01
python -u main.py --configs ${CONFIGS} \
                      --drop_last y \
                      --phase train \
                      --gathered n \
                      --loss_balance y \
                      --log_to_file n \
                      --backbone ${BACKBONE} \
                      --model_name ${MODEL_NAME} \
                      --gpu 3 4 5 6  \
                      --train_batch_size 8\
                      --val_batch_size 4 \
                      --memory_size ${MEMORY_SIZE}\
                      --projector "layer_2" "layer_3" "layer_4"\
                      --data_dir ${DATA_DIR} \
                      --loss_type ${LOSS_TYPE} \
                      --max_iters ${MAX_ITERS} \
                      --checkpoints_name ${CHECKPOINTS_NAME} \
                      --pretrained ${PRETRAINED_MODEL} \
                      --distributed \
                      --base_lr ${BASE_LR} \
                      2>&1 | tee ${LOG_FILE}


MODEL_NAME="resnet_fcn512_asp3_mep"
MEMORY_SIZE=16384
CHECKPOINTS_NAME="${MODEL_NAME}${MEMORY_SIZE}_${BACKBONE}_$(date +%F_%H-%M-%S)"
LOG_FILE="./experiment/log1/cityscapes/${CHECKPOINTS_NAME}.log"
echo "Logging to $LOG_FILE"
mkdir -p `dirname $LOG_FILE`
MAX_ITERS=80000
BASE_LR=0.01
python -u main.py --configs ${CONFIGS} \
                      --drop_last y \
                      --phase train \
                      --gathered n \
                      --loss_balance y \
                      --log_to_file n \
                      --backbone ${BACKBONE} \
                      --model_name ${MODEL_NAME} \
                      --gpu 3 4 5 6  \
                      --train_batch_size 8\
                      --val_batch_size 4 \
                      --memory_size ${MEMORY_SIZE}\
                      --projector "layer_2" "layer_3" "layer_4"\
                      --data_dir ${DATA_DIR} \
                      --loss_type ${LOSS_TYPE} \
                      --max_iters ${MAX_ITERS} \
                      --checkpoints_name ${CHECKPOINTS_NAME} \
                      --pretrained ${PRETRAINED_MODEL} \
                      --distributed \
                      --base_lr ${BASE_LR} \
                      2>&1 | tee ${LOG_FILE}

MODEL_NAME="resnet_fcn512_asp3_mep"
MEMORY_SIZE=16384
CHECKPOINTS_NAME="${MODEL_NAME}${MEMORY_SIZE}_${BACKBONE}_$(date +%F_%H-%M-%S)"
LOG_FILE="./experiment/log1/cityscapes/${CHECKPOINTS_NAME}.log"
echo "Logging to $LOG_FILE"
mkdir -p `dirname $LOG_FILE`
MAX_ITERS=80000
BASE_LR=0.02
python -u main.py --configs ${CONFIGS} \
                      --drop_last y \
                      --phase train \
                      --gathered n \
                      --loss_balance y \
                      --log_to_file n \
                      --backbone ${BACKBONE} \
                      --model_name ${MODEL_NAME} \
                      --gpu 3 4 5 6  \
                      --train_batch_size 8\
                      --val_batch_size 4 \
                      --memory_size ${MEMORY_SIZE}\
                      --projector "layer_2" "layer_3" "layer_4"\
                      --data_dir ${DATA_DIR} \
                      --loss_type ${LOSS_TYPE} \
                      --max_iters ${MAX_ITERS} \
                      --checkpoints_name ${CHECKPOINTS_NAME} \
                      --pretrained ${PRETRAINED_MODEL} \
                      --distributed \
                      --base_lr ${BASE_LR} \
                      2>&1 | tee ${LOG_FILE}

MEMORY_SIZE=16384
CHECKPOINTS_NAME="${MODEL_NAME}${MEMORY_SIZE}(183654)_${BACKBONE}_$(date +%F_%H-%M-%S)"
LOG_FILE="./experiment/log1/cityscapes/${CHECKPOINTS_NAME}.log"
echo "Logging to $LOG_FILE"
mkdir -p `dirname $LOG_FILE`
MAX_ITERS=80000
BASE_LR=0.02
python -u main.py --configs ${CONFIGS} \
                      --drop_last y \
                      --phase train \
                      --gathered n \
                      --loss_balance y \
                      --log_to_file n \
                      --backbone ${BACKBONE} \
                      --model_name ${MODEL_NAME} \
                      --gpu 3 4 5 6  \
                      --train_batch_size 8\
                      --val_batch_size 4 \
                      --memory_size ${MEMORY_SIZE}\
                      --projector "layer_2" "layer_3" "layer_4"\
                      --data_dir ${DATA_DIR} \
                      --loss_type ${LOSS_TYPE} \
                      --max_iters ${MAX_ITERS} \
                      --checkpoints_name ${CHECKPOINTS_NAME} \
                      --pretrained ${PRETRAINED_MODEL} \
                      --distributed \
                      --base_lr ${BASE_LR} \
                      2>&1 | tee ${LOG_FILE}