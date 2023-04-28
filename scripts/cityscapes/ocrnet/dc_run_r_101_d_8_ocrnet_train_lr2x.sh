#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd $SCRIPTPATH
cd ../../../
. config.profile
# check the enviroment info
# nvidia-smi
# ${PYTHON} -m pip install torchcontrib
# ${PYTHON} -m pip install git+https://github.com/lucasb-eyer/pydensecrf.git

# export PYTHONPATH="$PWD":$PYTHONPATH

DATA_DIR="../../input/openseg-cityscapes-gtfine"
SAVE_DIR="./result/cityscapes/checkpoints/"
BACKBONE="deepbase_resnet101_dilated8"

CONFIGS="configs/cityscapes/R_101_D_8.json"
CONFIGS_TEST="configs/cityscapes/R_101_D_8_TEST.json"

MODEL_NAME="spatial_ocrnet_dc"
LOSS_TYPE="fs_auxce_loss_dc"
CHECKPOINTS_NAME="dc_${MODEL_NAME}_${BACKBONE}_lr2x_$(date +%F_%H-%M-%S)"
LOG_FILE="./log/cityscapes/${CHECKPOINTS_NAME}.log"
echo "Logging to $LOG_FILE"
mkdir -p `dirname $LOG_FILE`

PRETRAINED_MODEL="../../input/pre-trained/resnet101-imagenet-openseg.pth"
MAX_ITERS=40000
BASE_LR=0.02

if [ "$1"x == "train"x ]; then
  python -u main.py --configs ${CONFIGS} \
                       --drop_last y \
                       --phase train \
                       --gathered n \
                       --loss_balance y \
                       --log_to_file n \
                       --backbone ${BACKBONE} \
                       --model_name ${MODEL_NAME} \
                       --workers 4\
                       --gpu 2 3 4 6 \
                       --train_batch_size 8\
                       --val_batch_size 4 \
                       --data_dir ${DATA_DIR} \
                       --loss_type ${LOSS_TYPE} \
                       --max_iters ${MAX_ITERS} \
                       --checkpoints_name ${CHECKPOINTS_NAME} \
                       --pretrained ${PRETRAINED_MODEL} \
                       --base_lr ${BASE_LR} \
                       --nbb_mult 1.0\
                       --distributed \
                       2>&1 | tee ${LOG_FILE}
                       

elif [ "$1"x == "resume"x ]; then
  python -u main.py --configs ${CONFIGS} \
                       --drop_last y \
                       --phase train \
                       --gathered n \
                       --loss_balance y \
                       --log_to_file n \
                       --backbone ${BACKBONE} \
                       --model_name ${MODEL_NAME} \
                       --max_iters ${MAX_ITERS} \
                       --data_dir ${DATA_DIR} \
                       --loss_type ${LOSS_TYPE} \
                       --gpu 0 1 2 3 \
                       --resume_continue y \
                       --resume ./checkpoints/cityscapes/${CHECKPOINTS_NAME}_latest.pth \
                       --checkpoints_name ${CHECKPOINTS_NAME} \
                        2>&1 | tee -a ${LOG_FILE}

elif [ "$1"x == "val"x ]; then
  ${PYTHON} -u main.py --configs ${CONFIGS} --drop_last y \
                       --backbone ${BACKBONE} --model_name ${MODEL_NAME} --checkpoints_name ${CHECKPOINTS_NAME} \
                       --phase test --gpu 0 1 2 3 --resume ./checkpoints/cityscapes/${CHECKPOINTS_NAME}_latest.pth \
                       --loss_type ${LOSS_TYPE} --test_dir ${DATA_DIR}/val/image \
                       --out_dir ${SAVE_DIR}${CHECKPOINTS_NAME}_val --data_dir ${DATA_DIR}


  cd lib/metrics
  ${PYTHON} -u cityscapes_evaluator.py --pred_dir ${SAVE_DIR}${CHECKPOINTS_NAME}_val/label  \
                                       --gt_dir ${DATA_DIR}/val/label

elif [ "$1"x == "segfix"x ]; then
  if [ "$3"x == "test"x ]; then
    DIR=${SAVE_DIR}${CHECKPOINTS_NAME}_test_ss/label
    echo "Applying SegFix for $DIR"
    ${PYTHON} scripts/cityscapes/segfix.py \
      --input $DIR \
      --split test \
      --offset ${DATA_ROOT}/cityscapes/test_offset/semantic/offset_hrnext/
  elif [ "$3"x == "val"x ]; then
    DIR=${SAVE_DIR}${CHECKPOINTS_NAME}_val/label
    echo "Applying SegFix for $DIR"
    ${PYTHON} scripts/cityscapes/segfix.py \
      --input $DIR \
      --split val \
      --offset ${DATA_ROOT}/cityscapes/val/offset_pred/semantic/offset_hrnext/
  fi

elif [ "$1"x == "test"x ]; then
  if [ "$3"x == "ss"x ]; then
    echo "[single scale] test"
    ${PYTHON} -u main.py --configs ${CONFIGS} --drop_last y \
                         --backbone ${BACKBONE} --model_name ${MODEL_NAME} --checkpoints_name ${CHECKPOINTS_NAME} \
                         --phase test --gpu 0 1 2 3 --resume ./checkpoints/cityscapes/${CHECKPOINTS_NAME}_latest.pth \
                         --test_dir ${DATA_DIR}/test --log_to_file n \
                         --out_dir ${SAVE_DIR}${CHECKPOINTS_NAME}_test_ss
  else
    echo "[multiple scale + flip] test"
    ${PYTHON} -u main.py --configs ${CONFIGS_TEST} --drop_last y \
                         --backbone ${BACKBONE} --model_name ${MODEL_NAME} --checkpoints_name ${CHECKPOINTS_NAME} \
                         --phase test --gpu 0 1 2 3 --resume ./checkpoints/cityscapes/${CHECKPOINTS_NAME}_latest.pth \
                         --test_dir ${DATA_DIR}/test --log_to_file n \
                         --out_dir ${SAVE_DIR}${CHECKPOINTS_NAME}_test_ms
  fi

else
  echo "$1"x" is invalid..."
fi
