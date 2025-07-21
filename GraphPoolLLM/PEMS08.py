import os
import sys
from easydict import EasyDict
sys.path.append(os.path.abspath(__file__ + '/../../..'))
from basicts.metrics import masked_mae, masked_rmse, masked_mape, mask_huber
from basicts.data import TimeSeriesForecastingDataset
from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings

from .arch import GraphLLMGatese
############################## Hot Parameters ##############################
# Dataset & Metrics configuration
DATA_NAME = 'PEMS08'  # Dataset name
regular_settings = get_regular_settings(DATA_NAME)
INPUT_LEN = regular_settings['INPUT_LEN']  # Length of input sequence
OUTPUT_LEN = regular_settings['OUTPUT_LEN']  # Length of output sequence
TRAIN_VAL_TEST_RATIO = regular_settings['TRAIN_VAL_TEST_RATIO']  # Train/Validation/Test split ratios
NORM_EACH_CHANNEL = regular_settings['NORM_EACH_CHANNEL'] # Whether to normalize each channel of the data
RESCALE = regular_settings['RESCALE'] # Whether to rescale the data
NULL_VAL = regular_settings['NULL_VAL'] # Null value in the data
# Model architecture and parameters
MODEL_ARCH = GraphLLMGatese
NUM_NODES = 170
MODEL_PARAM = {
    "task_name": 'short_term_forecast',
    "enc_in": INPUT_LEN,                        # num nodes
    "dec_in": INPUT_LEN,
    "c_out": OUTPUT_LEN,
    "seq_len": INPUT_LEN,
    "pred_len": INPUT_LEN,         # prediction sequence length
    "d_model": NUM_NODES,
    "d_layer": 1,
    "d_dimff": 64,
    "d_nhead": 4,
    "e_dimff": 64,
    "e_nhead": 8,
    "cross_nhead": 1,
    "d_ff": 152,
    "dropout": 0.1,  # 0.3 conv
    "freq": 'Bgatesel64-4-64-8-1-24-6', # B32-4-32-8-8-24-6
    "prompt_domain": 0.1,
    "llm_model": 'GPT2',# LLAMA, GPT2, BERT
    "llm_dim": 1280,# LLama7b:4096; GPT2-small:768; BERT-base:768; GPT2:1280
    "llm_layers": 24,
    "output_attention": False,
    "e_layer": 6,                            # [timeF, learned]
    "content": "This dataset contains traffic flow data on highways in San Bernardino over two months from July to August in 2016, with 170 detectors on 8 roads with a time interval of 5 minutes."
    }



NUM_EPOCHS = 50

############################## General Configuration ##############################
CFG = EasyDict()
# General settings
CFG.DESCRIPTION = 'An Example Config'
CFG.GPU_NUM = 1 # Number of GPUs to use (0 for CPU mode)
# Runner
CFG.RUNNER = SimpleTimeSeriesForecastingRunner

############################## Dataset Configuration ##############################
CFG.DATASET = EasyDict()
# Dataset settings
CFG.DATASET.NAME = DATA_NAME
CFG.DATASET.TYPE = TimeSeriesForecastingDataset
CFG.DATASET.PARAM = EasyDict({
    'dataset_name': DATA_NAME,
    'train_val_test_ratio': TRAIN_VAL_TEST_RATIO,
    'input_len': INPUT_LEN,
    'output_len': OUTPUT_LEN,
    # 'mode' is automatically set by the runner
})

############################## Scaler Configuration ##############################
CFG.SCALER = EasyDict()
# Scaler settings
CFG.SCALER.TYPE = ZScoreScaler # Scaler class
CFG.SCALER.PARAM = EasyDict({
    'dataset_name': DATA_NAME,
    'train_ratio': TRAIN_VAL_TEST_RATIO[0],
    'norm_each_channel': NORM_EACH_CHANNEL,
    'rescale': RESCALE,
})

############################## Model Configuration ##############################
CFG.MODEL = EasyDict()
# Model settings
CFG.MODEL.NAME = MODEL_ARCH.__name__
CFG.MODEL.ARCH = MODEL_ARCH
CFG.MODEL.PARAM = MODEL_PARAM
CFG.MODEL.FORWARD_FEATURES = [0, 1, 2]
CFG.MODEL.TARGET_FEATURES = [0]

############################## Metrics Configuration ##############################

CFG.METRICS = EasyDict()
# Metrics settings
CFG.METRICS.FUNCS = EasyDict({
                                'MAE': masked_mae,
                                'MAPE': masked_mape,
                                'RMSE': masked_rmse,
                            })
CFG.METRICS.TARGET = 'MAE'
CFG.METRICS.NULL_VAL = NULL_VAL

############################## Training Configuration ##############################
CFG.TRAIN = EasyDict()
CFG.TRAIN.NUM_EPOCHS = NUM_EPOCHS
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    'checkpoints',
    MODEL_ARCH.__name__,
    '_'.join([DATA_NAME, str(CFG.TRAIN.NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)])
)
CFG.TRAIN.LOSS = mask_huber
# Optimizer settings
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.001,
    "weight_decay": 0.0015
}
# Learning rate scheduler settings
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    "milestones": [25, 45],
    "gamma": 0.1
}

# Train data loader settings
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = 16
CFG.TRAIN.DATA.SHUFFLE = True

############################## Validation Configuration ##############################
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.BATCH_SIZE = 16

############################## Test Configuration ##############################
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.BATCH_SIZE = 16


############################## Evaluation Configuration ##############################

CFG.EVAL = EasyDict()

# Evaluation parameters
CFG.EVAL.HORIZONS = [3, 6, 12] # Prediction horizons for evaluation. Default: []
CFG.EVAL.USE_GPU = True # Whether to use GPU for evaluation. Default: True

'''
B8-4-64-8-1-24-6
2025-07-19 16:48:08,532 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 12.7155, Test MAPE: 8.4594, Test RMSE: 21.4978
2025-07-19 16:48:08,534 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 13.5069, Test MAPE: 8.9177, Test RMSE: 23.2624
2025-07-19 16:48:08,536 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 14.8536, Test MAPE: 9.8534, Test RMSE: 25.6317
2025-07-19 16:48:08,639 - easytorch-training - INFO - Result <test>: [test/time: 70.72 (s), test/loss: 13.0424, test/MAE: 13.5392, test/MAPE: 8.9747, test/RMSE: 23.2639]
2025-07-19 16:48:08,641 - easytorch-training - INFO - Test results saved to checkpoints/GraphLLM/PEMS08_50_12_12/33cd60fc8906d53d9766d1c998747cee/test_results.npz.
2025-07-19 16:48:08,641 - easytorch-training - INFO - Test metrics saved to checkpoints/GraphLLM/PEMS08_50_12_12/33cd60fc8906d53d9766d1c998747cee/test_metrics.json.

B32-4-32-8-8-24-6:
2025-07-18 19:25:25,121 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 12.6984, Test MAPE: 8.3411, Test RMSE: 21.5376
2025-07-18 19:25:25,123 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 13.5404, Test MAPE: 8.8584, Test RMSE: 23.3009
2025-07-18 19:25:25,124 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 14.8481, Test MAPE: 9.7899, Test RMSE: 25.6048
2025-07-18 19:25:25,249 - easytorch-training - INFO - Result <test>: [test/time: 71.21 (s), test/loss: 13.0423, test/MAE: 13.5369, test/MAPE: 8.8783, test/RMSE: 23.2685]
2025-07-18 19:25:25,251 - easytorch-training - INFO - Test results saved to checkpoints/GraphLLM/PEMS08_50_12_12/2c92a96788eda13f277c8368593b8e9e/test_results.npz.
2025-07-18 19:25:25,251 - easytorch-training - INFO - Test metrics saved to checkpoints/GraphLLM/PEMS08_50_12_12/2c92a96788eda13f277c8368593b8e9e/test_metrics.json.'''