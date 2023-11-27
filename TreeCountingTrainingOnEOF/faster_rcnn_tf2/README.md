# Tree counting

## Epel tree counting training model for SINARMAS

### Run
```
python train_tree_counting.py --help
usage: train_tree_counting.py [-h] [--pre_train PRE_TRAIN] --data_dir DATA_DIR
                              --log_dir LOG_DIR [--model_name MODEL_NAME]
                              [--model_size MODEL_SIZE] [--stage_1 STAGE_1]
                              [--stage_2 STAGE_2] [--stage_3 STAGE_3]
                              [--rps_ratio RPS_RATIO] [--rpn_ratio RPN_RATIO]
                              [--max_gt MAX_GT] [--train_step TRAIN_STEP]
                              [--val_step VAL_STEP] [--augment {yes,no}]

optional arguments:
  -h, --help            show this help message and exit
  --pre_train PRE_TRAIN
                        Pre train model path for transfer learning
  --data_dir DATA_DIR   Train data directory
  --log_dir LOG_DIR     Log directory for store model weight and log
  --model_name MODEL_NAME
                        Name of model
  --model_size MODEL_SIZE
                        Size of model
  --stage_1 STAGE_1     Num epoch for training network heads
  --stage_2 STAGE_2     Num epoch for fine tune layer Resnet stage 4 and up
  --stage_3 STAGE_3     Num epoch for fine tune all network layers
  --rps_ratio RPS_RATIO
                        Roi positive ratio
  --rpn_ratio RPN_RATIO
                        Rpn threshold ratio
  --max_gt MAX_GT       Max number ground truth instance each image
  --train_step TRAIN_STEP
                        Number train step for each epoch while training
  --val_step VAL_STEP   Number validate step for each epoch while training
  --augment {yes,no}    Choices Augmentation while training or not, must be
                        "yes" or "no"
```

#### Example
```bash
# Train tree counting model script
$ python train_tree_counting.py --data_dir='/path/to/data/dir' --log_dir='/path/to/logs' --stage_1=40 --stage_2=80 --stage_3=190 --model_name="test_model_16_08" --pre_train='/path/to/pretrainmodel.h5'
```