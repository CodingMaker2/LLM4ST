# Run a baseline model in BasicTS framework.
# pylint: disable=wrong-import-position
import os
import sys
from argparse import ArgumentParser

sys.path.append(os.path.abspath(__file__ + '/../..'))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import random
import numpy as np
import basicts

all_seed = 42
torch.manual_seed(all_seed)
torch.cuda.manual_seed(all_seed)
np.random.seed(all_seed)
random.seed(all_seed)
torch.backends.cudnn.deterministic = True
torch.set_num_threads(10) # aviod high cpu avg usage
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def parse_args():
    parser = ArgumentParser(description='Run time series forecasting model in BasicTS framework!')
    parser.add_argument('-c', '--cfg', default='baselines/GraphPoolLLM/METR-LA.py', help='training config')
    parser.add_argument('-g', '--gpus', default='0', help='visible gpus')
    return parser.parse_args()


def main():
    args = parse_args()
    basicts.launch_training(args.cfg, args.gpus, node_rank=0)


if __name__ == '__main__':
    main()

'''ssh://zzz2019@10.12.54.24:22/data/zzz2019/.conda/envs/BasicTS/bin/python3 -u /data2/zzz11/BasicTS-backup/experiments/train2.py
2025-06-15 18:28:48,768 - easytorch-launcher - INFO - Launching EasyTorch training.
DESCRIPTION: An Example Config
GPU_NUM: 1
RUNNER: <class 'basicts.runners.runner_zoo.simple_tsf_runner.SimpleTimeSeriesForecastingRunner'>
DATASET:
  NAME: METR-LA
  TYPE: <class 'basicts.data.simple_tsf_dataset_003.TimeSeriesForecastingDataset'>
  PARAM:
    dataset_name: METR-LA
    train_val_test_ratio: [0.7, 0.1, 0.2]
    input_len: 12
    output_len: 12
SCALER:
  TYPE: <class 'basicts.scaler.z_score_scaler.ZScoreScaler'>
  PARAM:
    dataset_name: METR-LA
    train_ratio: 0.7
    norm_each_channel: False
    rescale: True
MODEL:
  NAME: GraphLLM
  ARCH: <class 'baselines.GraphPoolLLM.arch.graphpoolllm_arch.GraphLLM'>
  PARAM:
    task_name: short_term_forecast
    enc_in: 12
    dec_in: 12
    c_out: 12
    seq_len: 12
    pred_len: 12
    d_model: 207
    n_heads: 2
    d_ff: 152
    dropout: 0.1
    freq: Rwv
    prompt_domain: 0
    llm_model: GPT2
    llm_dim: 1280
    llm_layers: 12
    output_attention: False
    embed: learned
    content: The dataset contains traffic speed data collected from 207 loop detectors on highways in Los Angeles County, aggregated in 5 minutes intervals over four months between March 2012 and June 2012.
  FORWARD_FEATURES: [0, 1, 2]
  TARGET_FEATURES: [0]
METRICS:
  FUNCS:
    MAE: masked_mae
    MAPE: masked_mape
    RMSE: masked_rmse
  TARGET: MAE
  NULL_VAL: 0.0
TRAIN:
  NUM_EPOCHS: 100
  CKPT_SAVE_DIR: checkpoints/GraphLLM/METR-LA_100_12_12
  LOSS: masked_mae
  OPTIM:
    TYPE: Adam
    PARAM:
      lr: 0.001
      weight_decay: 0.0003
  LR_SCHEDULER:
    TYPE: MultiStepLR
    PARAM:
      milestones: [20, 30]
      gamma: 0.1
  DATA:
    BATCH_SIZE: 16
    SHUFFLE: True
VAL:
  INTERVAL: 1
  DATA:
    BATCH_SIZE: 16
TEST:
  INTERVAL: 1
  DATA:
    BATCH_SIZE: 16
EVAL:
  HORIZONS: [3, 6, 12]
  USE_GPU: True

2025-06-15 18:28:49,249 - easytorch-env - INFO - Use devices 6.
2025-06-15 18:28:49,368 - easytorch-launcher - INFO - Initializing runner "<class 'basicts.runners.runner_zoo.simple_tsf_runner.SimpleTimeSeriesForecastingRunner'>"
2025-06-15 18:28:49,369 - easytorch-env - INFO - Disable TF32 mode
2025-06-15 18:28:49,369 - easytorch - INFO - Set ckpt save dir: 'checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab'
2025-06-15 18:28:49,369 - easytorch - INFO - Building model.
/data/zzz2019/.conda/envs/BasicTS/lib/python3.11/site-packages/huggingface_hub/file_download.py:896: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
  warnings.warn(
2025-06-15 18:30:11,761 - easytorch-training - INFO - Initializing training.
2025-06-15 18:30:11,762 - easytorch-training - INFO - Building training data loader.
2025-06-15 18:30:11,783 - easytorch-training - INFO - Train dataset length: 23968
2025-06-15 18:30:12,012 - easytorch-training - INFO - Set optim: Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    capturable: False
    differentiable: False
    eps: 1e-08
    foreach: None
    fused: None
    lr: 0.001
    maximize: False
    weight_decay: 0.0003
)
2025-06-15 18:30:12,013 - easytorch-training - INFO - Set lr_scheduler: <torch.optim.lr_scheduler.MultiStepLR object at 0x7fd7584c3650>
2025-06-15 18:30:12,014 - easytorch-training - INFO - Initializing validation.
2025-06-15 18:30:12,014 - easytorch-training - INFO - Building val data loader.
2025-06-15 18:30:12,050 - easytorch-training - INFO - Validation dataset length: 3404
2025-06-15 18:30:12,134 - easytorch-training - INFO - Test dataset length: 6831
2025-06-15 18:30:12,135 - easytorch-training - INFO - Number of parameters: 2358455
2025-06-15 18:30:12,136 - easytorch-training - INFO - Epoch 1 / 100
100%|███████████████████████████████████████| 1498/1498 [06:01<00:00,  4.14it/s]
2025-06-15 18:36:14,116 - easytorch-training - INFO - Result <train>: [train/time: 361.98 (s), train/lr: 1.00e-03, train/loss: 3.7662, train/MAE: 3.7662, train/MAPE: 10.8994, train/RMSE: 7.3284]
2025-06-15 18:36:14,116 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:41<00:00,  5.08it/s]
2025-06-15 18:36:56,035 - easytorch-training - INFO - Result <val>: [val/time: 41.92 (s), val/loss: 3.1101, val/MAE: 3.1101, val/MAPE: 9.0794, val/RMSE: 5.8398]
2025-06-15 18:36:57,178 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_best_val_MAE.pt saved
100%|█████████████████████████████████████████| 427/427 [01:23<00:00,  5.12it/s]
2025-06-15 18:38:20,606 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 3.0226, Test MAPE: 8.3151, Test RMSE: 5.9158
2025-06-15 18:38:20,608 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 3.5061, Test MAPE: 10.2707, Test RMSE: 7.1370
2025-06-15 18:38:20,610 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 4.1859, Test MAPE: 12.9222, Test RMSE: 8.6085
2025-06-15 18:38:20,633 - easytorch-training - INFO - Result <test>: [test/time: 83.45 (s), test/loss: 3.3177, test/MAE: 3.5068, test/MAPE: 10.2430, test/RMSE: 7.1770]
2025-06-15 18:38:21,754 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_001.pt saved
2025-06-15 18:38:21,754 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:06:14
2025-06-15 18:38:21,754 - easytorch-training - INFO - Epoch 2 / 100
100%|███████████████████████████████████████| 1498/1498 [06:00<00:00,  4.16it/s]
2025-06-15 18:44:22,223 - easytorch-training - INFO - Result <train>: [train/time: 360.47 (s), train/lr: 1.00e-03, train/loss: 3.2423, train/MAE: 3.2423, train/MAPE: 9.1025, train/RMSE: 6.4690]
2025-06-15 18:44:22,224 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:41<00:00,  5.11it/s]
2025-06-15 18:45:03,882 - easytorch-training - INFO - Result <val>: [val/time: 41.66 (s), val/loss: 2.9581, val/MAE: 2.9581, val/MAPE: 8.4906, val/RMSE: 5.5522]
2025-06-15 18:45:10,296 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_best_val_MAE.pt saved
100%|█████████████████████████████████████████| 427/427 [01:22<00:00,  5.15it/s]
2025-06-15 18:46:33,194 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.9382, Test MAPE: 7.9746, Test RMSE: 5.7549
2025-06-15 18:46:33,197 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 3.3592, Test MAPE: 9.6905, Test RMSE: 6.8174
2025-06-15 18:46:33,199 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.8126, Test MAPE: 11.5307, Test RMSE: 7.8577
2025-06-15 18:46:33,225 - easytorch-training - INFO - Result <test>: [test/time: 82.93 (s), test/loss: 3.1217, test/MAE: 3.3056, test/MAPE: 9.5096, test/RMSE: 6.7171]
2025-06-15 18:46:34,575 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_002.pt saved
2025-06-15 18:46:34,575 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:08:54
2025-06-15 18:46:34,575 - easytorch-training - INFO - Epoch 3 / 100
100%|███████████████████████████████████████| 1498/1498 [06:00<00:00,  4.16it/s]
2025-06-15 18:52:35,032 - easytorch-training - INFO - Result <train>: [train/time: 360.46 (s), train/lr: 1.00e-03, train/loss: 3.0926, train/MAE: 3.0926, train/MAPE: 8.5690, train/RMSE: 6.1787]
2025-06-15 18:52:35,033 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:41<00:00,  5.12it/s]
2025-06-15 18:53:16,663 - easytorch-training - INFO - Result <val>: [val/time: 41.63 (s), val/loss: 2.9413, val/MAE: 2.9413, val/MAPE: 8.2754, val/RMSE: 5.5076]
2025-06-15 18:53:23,291 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_best_val_MAE.pt saved
100%|█████████████████████████████████████████| 427/427 [01:22<00:00,  5.16it/s]
2025-06-15 18:54:46,054 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.9623, Test MAPE: 7.8558, Test RMSE: 5.7050
2025-06-15 18:54:46,056 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 3.3213, Test MAPE: 9.4395, Test RMSE: 6.7346
2025-06-15 18:54:46,059 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.7590, Test MAPE: 11.2897, Test RMSE: 7.6961
2025-06-15 18:54:46,083 - easytorch-training - INFO - Result <test>: [test/time: 82.79 (s), test/loss: 3.0983, test/MAE: 3.2865, test/MAPE: 9.3056, test/RMSE: 6.6258]
2025-06-15 18:54:47,391 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_003.pt saved
2025-06-15 18:54:47,391 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:09:47
2025-06-15 18:54:47,391 - easytorch-training - INFO - Epoch 4 / 100
100%|███████████████████████████████████████| 1498/1498 [06:00<00:00,  4.16it/s]
2025-06-15 19:00:47,626 - easytorch-training - INFO - Result <train>: [train/time: 360.24 (s), train/lr: 1.00e-03, train/loss: 3.0410, train/MAE: 3.0410, train/MAPE: 8.3851, train/RMSE: 6.1029]
2025-06-15 19:00:47,627 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:41<00:00,  5.11it/s]
2025-06-15 19:01:29,293 - easytorch-training - INFO - Result <val>: [val/time: 41.67 (s), val/loss: 2.8778, val/MAE: 2.8778, val/MAPE: 8.2485, val/RMSE: 5.3794]
2025-06-15 19:01:41,118 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_best_val_MAE.pt saved
100%|█████████████████████████████████████████| 427/427 [01:22<00:00,  5.16it/s]
2025-06-15 19:03:03,893 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.9007, Test MAPE: 7.7596, Test RMSE: 5.5813
2025-06-15 19:03:03,896 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 3.2649, Test MAPE: 9.3657, Test RMSE: 6.5998
2025-06-15 19:03:03,898 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.6525, Test MAPE: 11.0961, Test RMSE: 7.5401
2025-06-15 19:03:03,920 - easytorch-training - INFO - Result <test>: [test/time: 82.80 (s), test/loss: 3.0322, test/MAE: 3.2148, test/MAPE: 9.2047, test/RMSE: 6.4941]
2025-06-15 19:03:05,211 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_004.pt saved
2025-06-15 19:03:05,211 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:12:19
2025-06-15 19:03:05,211 - easytorch-training - INFO - Epoch 5 / 100
100%|███████████████████████████████████████| 1498/1498 [06:00<00:00,  4.16it/s]
2025-06-15 19:09:05,633 - easytorch-training - INFO - Result <train>: [train/time: 360.42 (s), train/lr: 1.00e-03, train/loss: 2.9980, train/MAE: 2.9980, train/MAPE: 8.2276, train/RMSE: 6.0228]
2025-06-15 19:09:05,635 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:41<00:00,  5.12it/s]
2025-06-15 19:09:47,272 - easytorch-training - INFO - Result <val>: [val/time: 41.64 (s), val/loss: 2.8732, val/MAE: 2.8732, val/MAPE: 8.0328, val/RMSE: 5.4070]
2025-06-15 19:09:58,905 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_best_val_MAE.pt saved
100%|█████████████████████████████████████████| 427/427 [01:22<00:00,  5.16it/s]
2025-06-15 19:11:21,659 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.8640, Test MAPE: 7.5817, Test RMSE: 5.5564
2025-06-15 19:11:21,661 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 3.2314, Test MAPE: 9.1180, Test RMSE: 6.5933
2025-06-15 19:11:21,662 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.6248, Test MAPE: 10.5840, Test RMSE: 7.5314
2025-06-15 19:11:21,685 - easytorch-training - INFO - Result <test>: [test/time: 82.78 (s), test/loss: 2.9969, test/MAE: 3.1804, test/MAPE: 8.8888, test/RMSE: 6.4719]
2025-06-15 19:11:22,960 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_005.pt saved
2025-06-15 19:11:22,961 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:13:48
2025-06-15 19:11:22,961 - easytorch-training - INFO - Epoch 6 / 100
100%|███████████████████████████████████████| 1498/1498 [06:00<00:00,  4.16it/s]
2025-06-15 19:17:23,038 - easytorch-training - INFO - Result <train>: [train/time: 360.08 (s), train/lr: 1.00e-03, train/loss: 2.9671, train/MAE: 2.9671, train/MAPE: 8.1332, train/RMSE: 5.9647]
2025-06-15 19:17:23,040 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:41<00:00,  5.12it/s]
2025-06-15 19:18:04,677 - easytorch-training - INFO - Result <val>: [val/time: 41.64 (s), val/loss: 2.8376, val/MAE: 2.8376, val/MAPE: 7.9425, val/RMSE: 5.3570]
2025-06-15 19:18:16,419 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_best_val_MAE.pt saved
100%|█████████████████████████████████████████| 427/427 [01:22<00:00,  5.16it/s]
2025-06-15 19:19:39,165 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.8030, Test MAPE: 7.4683, Test RMSE: 5.4321
2025-06-15 19:19:39,167 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 3.1746, Test MAPE: 8.9451, Test RMSE: 6.4507
2025-06-15 19:19:39,169 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.5943, Test MAPE: 10.7240, Test RMSE: 7.4653
2025-06-15 19:19:39,192 - easytorch-training - INFO - Result <test>: [test/time: 82.77 (s), test/loss: 2.9555, test/MAE: 3.1327, test/MAPE: 8.8483, test/RMSE: 6.3729]
2025-06-15 19:19:40,456 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_006.pt saved
2025-06-15 19:19:40,457 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:14:44
2025-06-15 19:19:40,457 - easytorch-training - INFO - Epoch 7 / 100
100%|███████████████████████████████████████| 1498/1498 [06:00<00:00,  4.16it/s]
2025-06-15 19:25:40,665 - easytorch-training - INFO - Result <train>: [train/time: 360.21 (s), train/lr: 1.00e-03, train/loss: 2.9384, train/MAE: 2.9384, train/MAPE: 8.0150, train/RMSE: 5.8839]
2025-06-15 19:25:40,665 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:41<00:00,  5.12it/s]
2025-06-15 19:26:22,292 - easytorch-training - INFO - Result <val>: [val/time: 41.63 (s), val/loss: 2.8683, val/MAE: 2.8683, val/MAPE: 8.1281, val/RMSE: 5.3009]
100%|█████████████████████████████████████████| 427/427 [01:22<00:00,  5.15it/s]
2025-06-15 19:27:45,165 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.8324, Test MAPE: 7.5696, Test RMSE: 5.4515
2025-06-15 19:27:45,167 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 3.2072, Test MAPE: 9.0433, Test RMSE: 6.3532
2025-06-15 19:27:45,169 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.6520, Test MAPE: 10.9193, Test RMSE: 7.2966
2025-06-15 19:27:45,195 - easytorch-training - INFO - Result <test>: [test/time: 82.90 (s), test/loss: 2.9904, test/MAE: 3.1683, test/MAPE: 8.9422, test/RMSE: 6.2713]
2025-06-15 19:27:46,543 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_007.pt saved
2025-06-15 19:27:46,544 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:12:40
2025-06-15 19:27:46,544 - easytorch-training - INFO - Epoch 8 / 100
100%|███████████████████████████████████████| 1498/1498 [06:00<00:00,  4.16it/s]
2025-06-15 19:33:46,733 - easytorch-training - INFO - Result <train>: [train/time: 360.19 (s), train/lr: 1.00e-03, train/loss: 2.9019, train/MAE: 2.9019, train/MAPE: 7.8864, train/RMSE: 5.8063]
2025-06-15 19:33:46,734 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:41<00:00,  5.12it/s]
2025-06-15 19:34:28,354 - easytorch-training - INFO - Result <val>: [val/time: 41.62 (s), val/loss: 2.8458, val/MAE: 2.8458, val/MAPE: 8.2041, val/RMSE: 5.2834]
100%|█████████████████████████████████████████| 427/427 [01:22<00:00,  5.15it/s]
2025-06-15 19:35:51,202 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.8288, Test MAPE: 7.7560, Test RMSE: 5.3675
2025-06-15 19:35:51,204 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 3.1461, Test MAPE: 9.1744, Test RMSE: 6.3608
2025-06-15 19:35:51,206 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.5415, Test MAPE: 10.8234, Test RMSE: 7.3482
2025-06-15 19:35:51,231 - easytorch-training - INFO - Result <test>: [test/time: 82.88 (s), test/loss: 2.9400, test/MAE: 3.1189, test/MAPE: 9.0448, test/RMSE: 6.2843]
2025-06-15 19:35:52,506 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_008.pt saved
2025-06-15 19:35:52,506 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:11:06
2025-06-15 19:35:52,506 - easytorch-training - INFO - Epoch 9 / 100
100%|███████████████████████████████████████| 1498/1498 [06:00<00:00,  4.16it/s]
2025-06-15 19:41:52,653 - easytorch-training - INFO - Result <train>: [train/time: 360.15 (s), train/lr: 1.00e-03, train/loss: 2.8735, train/MAE: 2.8735, train/MAPE: 7.7724, train/RMSE: 5.7342]
2025-06-15 19:41:52,655 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:41<00:00,  5.12it/s]
2025-06-15 19:42:34,285 - easytorch-training - INFO - Result <val>: [val/time: 41.63 (s), val/loss: 2.7513, val/MAE: 2.7513, val/MAPE: 7.7297, val/RMSE: 5.2331]
2025-06-15 19:42:40,727 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_best_val_MAE.pt saved
100%|█████████████████████████████████████████| 427/427 [01:22<00:00,  5.16it/s]
2025-06-15 19:44:03,532 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.7458, Test MAPE: 7.2865, Test RMSE: 5.3467
2025-06-15 19:44:03,534 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 3.0800, Test MAPE: 8.6702, Test RMSE: 6.3186
2025-06-15 19:44:03,536 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.4725, Test MAPE: 10.3548, Test RMSE: 7.2947
2025-06-15 19:44:03,560 - easytorch-training - INFO - Result <test>: [test/time: 82.83 (s), test/loss: 2.8683, test/MAE: 3.0415, test/MAPE: 8.5632, test/RMSE: 6.2285]
2025-06-15 19:44:04,852 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_009.pt saved
2025-06-15 19:44:04,852 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:11:04
2025-06-15 19:44:04,852 - easytorch-training - INFO - Epoch 10 / 100
100%|███████████████████████████████████████| 1498/1498 [06:00<00:00,  4.16it/s]
2025-06-15 19:50:05,075 - easytorch-training - INFO - Result <train>: [train/time: 360.22 (s), train/lr: 1.00e-03, train/loss: 2.8472, train/MAE: 2.8472, train/MAPE: 7.6838, train/RMSE: 5.6772]
2025-06-15 19:50:05,076 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:41<00:00,  5.12it/s]
2025-06-15 19:50:46,710 - easytorch-training - INFO - Result <val>: [val/time: 41.63 (s), val/loss: 2.7796, val/MAE: 2.7796, val/MAPE: 7.6745, val/RMSE: 5.2045]
100%|█████████████████████████████████████████| 427/427 [01:22<00:00,  5.15it/s]
2025-06-15 19:52:09,594 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.7502, Test MAPE: 7.2996, Test RMSE: 5.2472
2025-06-15 19:52:09,596 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 3.0764, Test MAPE: 8.5325, Test RMSE: 6.2006
2025-06-15 19:52:09,598 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.5142, Test MAPE: 10.1720, Test RMSE: 7.2258
2025-06-15 19:52:09,620 - easytorch-training - INFO - Result <test>: [test/time: 82.91 (s), test/loss: 2.8866, test/MAE: 3.0536, test/MAPE: 8.4634, test/RMSE: 6.1506]
2025-06-15 19:52:10,756 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_010.pt saved
2025-06-15 19:52:11,932 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:10:10
2025-06-15 19:52:11,932 - easytorch-training - INFO - Epoch 11 / 100
100%|███████████████████████████████████████| 1498/1498 [06:01<00:00,  4.15it/s]
2025-06-15 19:58:13,021 - easytorch-training - INFO - Result <train>: [train/time: 361.09 (s), train/lr: 1.00e-03, train/loss: 2.8300, train/MAE: 2.8300, train/MAPE: 7.6194, train/RMSE: 5.6353]
2025-06-15 19:58:13,022 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:41<00:00,  5.09it/s]
2025-06-15 19:58:54,837 - easytorch-training - INFO - Result <val>: [val/time: 41.81 (s), val/loss: 2.7302, val/MAE: 2.7302, val/MAPE: 7.4046, val/RMSE: 5.1503]
2025-06-15 19:59:01,511 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_best_val_MAE.pt saved
100%|█████████████████████████████████████████| 427/427 [01:23<00:00,  5.13it/s]
2025-06-15 20:00:24,722 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.7020, Test MAPE: 6.9171, Test RMSE: 5.1478
2025-06-15 20:00:24,724 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 3.0246, Test MAPE: 8.2077, Test RMSE: 6.0560
2025-06-15 20:00:24,726 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.4229, Test MAPE: 9.7648, Test RMSE: 7.0763
2025-06-15 20:00:24,748 - easytorch-training - INFO - Result <test>: [test/time: 83.24 (s), test/loss: 2.8261, test/MAE: 2.9965, test/MAPE: 8.1165, test/RMSE: 6.0224]
2025-06-15 20:00:25,993 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_011.pt saved
2025-06-15 20:00:25,994 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:10:29
2025-06-15 20:00:25,994 - easytorch-training - INFO - Epoch 12 / 100
100%|███████████████████████████████████████| 1498/1498 [06:03<00:00,  4.12it/s]
2025-06-15 20:06:29,244 - easytorch-training - INFO - Result <train>: [train/time: 363.25 (s), train/lr: 1.00e-03, train/loss: 2.8128, train/MAE: 2.8128, train/MAPE: 7.5553, train/RMSE: 5.5957]
2025-06-15 20:06:29,245 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:42<00:00,  5.05it/s]
2025-06-15 20:07:11,460 - easytorch-training - INFO - Result <val>: [val/time: 42.22 (s), val/loss: 2.7172, val/MAE: 2.7172, val/MAPE: 7.5826, val/RMSE: 5.1174]
2025-06-15 20:07:19,280 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_best_val_MAE.pt saved
100%|█████████████████████████████████████████| 427/427 [01:23<00:00,  5.09it/s]
2025-06-15 20:08:43,252 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6778, Test MAPE: 6.9941, Test RMSE: 5.1265
2025-06-15 20:08:43,255 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 3.0021, Test MAPE: 8.4263, Test RMSE: 6.0748
2025-06-15 20:08:43,258 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3925, Test MAPE: 10.0204, Test RMSE: 7.0709
2025-06-15 20:08:43,285 - easytorch-training - INFO - Result <test>: [test/time: 84.00 (s), test/loss: 2.7988, test/MAE: 2.9704, test/MAPE: 8.2907, test/RMSE: 6.0241]
2025-06-15 20:08:44,600 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_012.pt saved
2025-06-15 20:08:44,601 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:11:22
2025-06-15 20:08:44,601 - easytorch-training - INFO - Epoch 13 / 100
100%|███████████████████████████████████████| 1498/1498 [06:05<00:00,  4.10it/s]
2025-06-15 20:14:49,616 - easytorch-training - INFO - Result <train>: [train/time: 365.01 (s), train/lr: 1.00e-03, train/loss: 2.7979, train/MAE: 2.7979, train/MAPE: 7.5013, train/RMSE: 5.5613]
2025-06-15 20:14:49,617 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:42<00:00,  5.05it/s]
2025-06-15 20:15:31,835 - easytorch-training - INFO - Result <val>: [val/time: 42.22 (s), val/loss: 2.7489, val/MAE: 2.7489, val/MAPE: 7.4483, val/RMSE: 5.1500]
100%|█████████████████████████████████████████| 427/427 [01:24<00:00,  5.08it/s]
2025-06-15 20:16:55,914 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.7054, Test MAPE: 6.9763, Test RMSE: 5.1868
2025-06-15 20:16:55,916 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 3.0480, Test MAPE: 8.3290, Test RMSE: 6.1375
2025-06-15 20:16:55,919 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.4802, Test MAPE: 10.0925, Test RMSE: 7.2169
2025-06-15 20:16:55,940 - easytorch-training - INFO - Result <test>: [test/time: 84.10 (s), test/loss: 2.8533, test/MAE: 3.0216, test/MAPE: 8.2626, test/RMSE: 6.1097]
2025-06-15 20:16:57,158 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_013.pt saved
2025-06-15 20:16:57,159 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:11:21
2025-06-15 20:16:57,159 - easytorch-training - INFO - Epoch 14 / 100
100%|███████████████████████████████████████| 1498/1498 [06:04<00:00,  4.11it/s]
2025-06-15 20:23:01,819 - easytorch-training - INFO - Result <train>: [train/time: 364.66 (s), train/lr: 1.00e-03, train/loss: 2.7875, train/MAE: 2.7875, train/MAPE: 7.4650, train/RMSE: 5.5352]
2025-06-15 20:23:01,820 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:42<00:00,  5.04it/s]
2025-06-15 20:23:44,063 - easytorch-training - INFO - Result <val>: [val/time: 42.24 (s), val/loss: 2.7006, val/MAE: 2.7006, val/MAPE: 7.3321, val/RMSE: 5.1178]
2025-06-15 20:23:55,845 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_best_val_MAE.pt saved
100%|█████████████████████████████████████████| 427/427 [01:23<00:00,  5.10it/s]
2025-06-15 20:25:19,602 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6619, Test MAPE: 6.9053, Test RMSE: 5.1380
2025-06-15 20:25:19,605 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9898, Test MAPE: 8.1956, Test RMSE: 6.0609
2025-06-15 20:25:19,607 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3962, Test MAPE: 9.8008, Test RMSE: 7.1377
2025-06-15 20:25:19,629 - easytorch-training - INFO - Result <test>: [test/time: 83.78 (s), test/loss: 2.7926, test/MAE: 2.9613, test/MAPE: 8.0891, test/RMSE: 6.0392]
2025-06-15 20:25:21,009 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_014.pt saved
2025-06-15 20:25:21,009 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:12:41
2025-06-15 20:25:21,009 - easytorch-training - INFO - Epoch 15 / 100
100%|███████████████████████████████████████| 1498/1498 [06:04<00:00,  4.11it/s]
2025-06-15 20:31:25,716 - easytorch-training - INFO - Result <train>: [train/time: 364.71 (s), train/lr: 1.00e-03, train/loss: 2.7812, train/MAE: 2.7812, train/MAPE: 7.4411, train/RMSE: 5.5241]
2025-06-15 20:31:25,717 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:42<00:00,  5.05it/s]
2025-06-15 20:32:07,907 - easytorch-training - INFO - Result <val>: [val/time: 42.19 (s), val/loss: 2.7406, val/MAE: 2.7406, val/MAPE: 7.6011, val/RMSE: 5.1207]
100%|█████████████████████████████████████████| 427/427 [01:23<00:00,  5.09it/s]
2025-06-15 20:33:31,827 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6786, Test MAPE: 7.0055, Test RMSE: 5.1029
2025-06-15 20:33:31,829 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 3.0285, Test MAPE: 8.3469, Test RMSE: 6.0705
2025-06-15 20:33:31,831 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.4412, Test MAPE: 10.1845, Test RMSE: 7.1383
2025-06-15 20:33:31,855 - easytorch-training - INFO - Result <test>: [test/time: 83.95 (s), test/loss: 2.8277, test/MAE: 2.9929, test/MAPE: 8.2974, test/RMSE: 6.0328]
2025-06-15 20:33:33,223 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_015.pt saved
2025-06-15 20:33:33,223 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:12:32
2025-06-15 20:33:33,223 - easytorch-training - INFO - Epoch 16 / 100
100%|███████████████████████████████████████| 1498/1498 [06:05<00:00,  4.10it/s]
2025-06-15 20:39:38,314 - easytorch-training - INFO - Result <train>: [train/time: 365.09 (s), train/lr: 1.00e-03, train/loss: 2.7662, train/MAE: 2.7662, train/MAPE: 7.3797, train/RMSE: 5.4824]
2025-06-15 20:39:38,316 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:42<00:00,  5.04it/s]
2025-06-15 20:40:20,553 - easytorch-training - INFO - Result <val>: [val/time: 42.24 (s), val/loss: 2.6936, val/MAE: 2.6936, val/MAPE: 7.4083, val/RMSE: 5.0999]
2025-06-15 20:40:27,662 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_best_val_MAE.pt saved
100%|█████████████████████████████████████████| 427/427 [01:23<00:00,  5.10it/s]
2025-06-15 20:41:51,440 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6541, Test MAPE: 6.9578, Test RMSE: 5.1043
2025-06-15 20:41:51,443 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9747, Test MAPE: 8.2251, Test RMSE: 6.0119
2025-06-15 20:41:51,446 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3677, Test MAPE: 9.7869, Test RMSE: 7.0139
2025-06-15 20:41:51,472 - easytorch-training - INFO - Result <test>: [test/time: 83.81 (s), test/loss: 2.7781, test/MAE: 2.9453, test/MAPE: 8.1252, test/RMSE: 5.9713]
2025-06-15 20:41:52,839 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_016.pt saved
2025-06-15 20:41:52,839 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:13:11
2025-06-15 20:41:52,839 - easytorch-training - INFO - Epoch 17 / 100
100%|███████████████████████████████████████| 1498/1498 [06:05<00:00,  4.10it/s]
2025-06-15 20:47:57,994 - easytorch-training - INFO - Result <train>: [train/time: 365.15 (s), train/lr: 1.00e-03, train/loss: 2.7610, train/MAE: 2.7610, train/MAPE: 7.3707, train/RMSE: 5.4767]
2025-06-15 20:47:57,995 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:42<00:00,  5.04it/s]
2025-06-15 20:48:40,252 - easytorch-training - INFO - Result <val>: [val/time: 42.26 (s), val/loss: 2.6899, val/MAE: 2.6899, val/MAPE: 7.2884, val/RMSE: 5.0825]
2025-06-15 20:48:47,604 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_best_val_MAE.pt saved
100%|█████████████████████████████████████████| 427/427 [01:23<00:00,  5.09it/s]
2025-06-15 20:50:11,565 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6525, Test MAPE: 6.8380, Test RMSE: 5.0913
2025-06-15 20:50:11,568 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9796, Test MAPE: 8.1577, Test RMSE: 6.0181
2025-06-15 20:50:11,570 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3639, Test MAPE: 9.6221, Test RMSE: 7.0356
2025-06-15 20:50:11,593 - easytorch-training - INFO - Result <test>: [test/time: 83.99 (s), test/loss: 2.7789, test/MAE: 2.9457, test/MAPE: 8.0090, test/RMSE: 5.9812]
2025-06-15 20:50:12,863 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_017.pt saved
2025-06-15 20:50:12,864 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:13:48
2025-06-15 20:50:12,864 - easytorch-training - INFO - Epoch 18 / 100
100%|███████████████████████████████████████| 1498/1498 [06:05<00:00,  4.10it/s]
2025-06-15 20:56:18,074 - easytorch-training - INFO - Result <train>: [train/time: 365.21 (s), train/lr: 1.00e-03, train/loss: 2.7513, train/MAE: 2.7513, train/MAPE: 7.3301, train/RMSE: 5.4516]
2025-06-15 20:56:18,075 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:42<00:00,  5.03it/s]
2025-06-15 20:57:00,413 - easytorch-training - INFO - Result <val>: [val/time: 42.34 (s), val/loss: 2.7133, val/MAE: 2.7133, val/MAPE: 7.4545, val/RMSE: 5.1392]
100%|█████████████████████████████████████████| 427/427 [01:24<00:00,  5.08it/s]
2025-06-15 20:58:24,489 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6719, Test MAPE: 6.8739, Test RMSE: 5.1143
2025-06-15 20:58:24,491 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 3.0185, Test MAPE: 8.1882, Test RMSE: 6.0870
2025-06-15 20:58:24,493 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3938, Test MAPE: 9.8522, Test RMSE: 7.0657
2025-06-15 20:58:24,515 - easytorch-training - INFO - Result <test>: [test/time: 84.10 (s), test/loss: 2.8057, test/MAE: 2.9777, test/MAPE: 8.1161, test/RMSE: 6.0256]
2025-06-15 20:58:25,802 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_018.pt saved
2025-06-15 20:58:25,802 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:13:41
2025-06-15 20:58:25,802 - easytorch-training - INFO - Epoch 19 / 100
100%|███████████████████████████████████████| 1498/1498 [06:05<00:00,  4.10it/s]
2025-06-15 21:04:31,336 - easytorch-training - INFO - Result <train>: [train/time: 365.53 (s), train/lr: 1.00e-03, train/loss: 2.7475, train/MAE: 2.7475, train/MAPE: 7.3141, train/RMSE: 5.4435]
2025-06-15 21:04:31,337 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:42<00:00,  5.03it/s]
2025-06-15 21:05:13,672 - easytorch-training - INFO - Result <val>: [val/time: 42.33 (s), val/loss: 2.7017, val/MAE: 2.7017, val/MAPE: 7.4300, val/RMSE: 5.0995]
100%|█████████████████████████████████████████| 427/427 [01:24<00:00,  5.07it/s]
2025-06-15 21:06:37,918 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6546, Test MAPE: 6.9156, Test RMSE: 5.0745
2025-06-15 21:06:37,921 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9936, Test MAPE: 8.1776, Test RMSE: 6.0463
2025-06-15 21:06:37,923 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3924, Test MAPE: 9.8194, Test RMSE: 7.1091
2025-06-15 21:06:37,949 - easytorch-training - INFO - Result <test>: [test/time: 84.28 (s), test/loss: 2.7945, test/MAE: 2.9601, test/MAPE: 8.1151, test/RMSE: 6.0040]
2025-06-15 21:06:39,205 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_019.pt saved
2025-06-15 21:06:39,206 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:13:37
2025-06-15 21:06:39,206 - easytorch-training - INFO - Epoch 20 / 100
100%|███████████████████████████████████████| 1498/1498 [06:05<00:00,  4.10it/s]
2025-06-15 21:12:44,606 - easytorch-training - INFO - Result <train>: [train/time: 365.40 (s), train/lr: 1.00e-03, train/loss: 2.7396, train/MAE: 2.7396, train/MAPE: 7.2885, train/RMSE: 5.4275]
2025-06-15 21:12:44,607 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:42<00:00,  5.04it/s]
2025-06-15 21:13:26,895 - easytorch-training - INFO - Result <val>: [val/time: 42.29 (s), val/loss: 2.6956, val/MAE: 2.6956, val/MAPE: 7.4137, val/RMSE: 5.0978]
100%|█████████████████████████████████████████| 427/427 [01:24<00:00,  5.07it/s]
2025-06-15 21:14:51,199 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6654, Test MAPE: 6.9157, Test RMSE: 5.1222
2025-06-15 21:14:51,201 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9919, Test MAPE: 8.2464, Test RMSE: 6.0696
2025-06-15 21:14:51,203 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3816, Test MAPE: 9.8765, Test RMSE: 7.0700
2025-06-15 21:14:51,225 - easytorch-training - INFO - Result <test>: [test/time: 84.33 (s), test/loss: 2.7877, test/MAE: 2.9573, test/MAPE: 8.1457, test/RMSE: 6.0093]
2025-06-15 21:14:52,509 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_020.pt saved
2025-06-15 21:14:53,825 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:13:40
2025-06-15 21:14:53,825 - easytorch-training - INFO - Epoch 21 / 100
100%|███████████████████████████████████████| 1498/1498 [06:05<00:00,  4.10it/s]
2025-06-15 21:20:59,398 - easytorch-training - INFO - Result <train>: [train/time: 365.57 (s), train/lr: 1.00e-04, train/loss: 2.6606, train/MAE: 2.6606, train/MAPE: 7.0231, train/RMSE: 5.2625]
2025-06-15 21:20:59,398 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:42<00:00,  5.03it/s]
2025-06-15 21:21:41,748 - easytorch-training - INFO - Result <val>: [val/time: 42.35 (s), val/loss: 2.6648, val/MAE: 2.6648, val/MAPE: 7.2946, val/RMSE: 5.0625]
2025-06-15 21:21:48,479 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_best_val_MAE.pt saved
100%|█████████████████████████████████████████| 427/427 [01:24<00:00,  5.08it/s]
2025-06-15 21:23:12,619 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6229, Test MAPE: 6.8068, Test RMSE: 5.0315
2025-06-15 21:23:12,622 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9542, Test MAPE: 8.1205, Test RMSE: 5.9882
2025-06-15 21:23:12,625 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3345, Test MAPE: 9.6752, Test RMSE: 7.0139
2025-06-15 21:23:12,648 - easytorch-training - INFO - Result <test>: [test/time: 84.17 (s), test/loss: 2.7520, test/MAE: 2.9186, test/MAPE: 8.0106, test/RMSE: 5.9459]
2025-06-15 21:23:13,924 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_021.pt saved
2025-06-15 21:23:13,925 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:14:09
2025-06-15 21:23:13,925 - easytorch-training - INFO - Epoch 22 / 100
100%|███████████████████████████████████████| 1498/1498 [06:05<00:00,  4.10it/s]
2025-06-15 21:29:19,486 - easytorch-training - INFO - Result <train>: [train/time: 365.56 (s), train/lr: 1.00e-04, train/loss: 2.6456, train/MAE: 2.6456, train/MAPE: 6.9734, train/RMSE: 5.2276]
2025-06-15 21:29:19,487 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:42<00:00,  5.02it/s]
2025-06-15 21:30:01,892 - easytorch-training - INFO - Result <val>: [val/time: 42.40 (s), val/loss: 2.6662, val/MAE: 2.6662, val/MAPE: 7.3104, val/RMSE: 5.0569]
100%|█████████████████████████████████████████| 427/427 [01:24<00:00,  5.08it/s]
2025-06-15 21:31:26,036 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6221, Test MAPE: 6.7921, Test RMSE: 5.0169
2025-06-15 21:31:26,038 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9587, Test MAPE: 8.1404, Test RMSE: 5.9853
2025-06-15 21:31:26,040 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3474, Test MAPE: 9.7894, Test RMSE: 7.0423
2025-06-15 21:31:26,063 - easytorch-training - INFO - Result <test>: [test/time: 84.17 (s), test/loss: 2.7561, test/MAE: 2.9232, test/MAPE: 8.0437, test/RMSE: 5.9501]
2025-06-15 21:31:27,348 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_022.pt saved
2025-06-15 21:31:27,348 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:14:04
2025-06-15 21:31:27,348 - easytorch-training - INFO - Epoch 23 / 100
100%|███████████████████████████████████████| 1498/1498 [06:05<00:00,  4.10it/s]
2025-06-15 21:37:32,849 - easytorch-training - INFO - Result <train>: [train/time: 365.50 (s), train/lr: 1.00e-04, train/loss: 2.6386, train/MAE: 2.6386, train/MAPE: 6.9461, train/RMSE: 5.2106]
2025-06-15 21:37:32,850 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:42<00:00,  5.04it/s]
2025-06-15 21:38:15,120 - easytorch-training - INFO - Result <val>: [val/time: 42.27 (s), val/loss: 2.6709, val/MAE: 2.6709, val/MAPE: 7.2909, val/RMSE: 5.0743]
100%|█████████████████████████████████████████| 427/427 [01:24<00:00,  5.08it/s]
2025-06-15 21:39:39,134 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6236, Test MAPE: 6.7493, Test RMSE: 5.0312
2025-06-15 21:39:39,137 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9572, Test MAPE: 8.0750, Test RMSE: 5.9998
2025-06-15 21:39:39,140 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3442, Test MAPE: 9.7430, Test RMSE: 7.0454
2025-06-15 21:39:39,165 - easytorch-training - INFO - Result <test>: [test/time: 84.04 (s), test/loss: 2.7545, test/MAE: 2.9221, test/MAPE: 7.9885, test/RMSE: 5.9572]
2025-06-15 21:39:40,491 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_023.pt saved
2025-06-15 21:39:40,492 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:13:59
2025-06-15 21:39:40,492 - easytorch-training - INFO - Epoch 24 / 100
100%|███████████████████████████████████████| 1498/1498 [06:04<00:00,  4.11it/s]
2025-06-15 21:45:45,103 - easytorch-training - INFO - Result <train>: [train/time: 364.61 (s), train/lr: 1.00e-04, train/loss: 2.6332, train/MAE: 2.6332, train/MAPE: 6.9262, train/RMSE: 5.1959]
2025-06-15 21:45:45,104 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:42<00:00,  5.06it/s]
2025-06-15 21:46:27,230 - easytorch-training - INFO - Result <val>: [val/time: 42.13 (s), val/loss: 2.6622, val/MAE: 2.6622, val/MAPE: 7.2461, val/RMSE: 5.0549]
2025-06-15 21:46:39,180 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_best_val_MAE.pt saved
100%|█████████████████████████████████████████| 427/427 [01:23<00:00,  5.12it/s]
2025-06-15 21:48:02,628 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6179, Test MAPE: 6.7740, Test RMSE: 5.0166
2025-06-15 21:48:02,631 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9518, Test MAPE: 8.0551, Test RMSE: 5.9820
2025-06-15 21:48:02,633 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3345, Test MAPE: 9.5888, Test RMSE: 7.0154
2025-06-15 21:48:02,655 - easytorch-training - INFO - Result <test>: [test/time: 83.47 (s), test/loss: 2.7478, test/MAE: 2.9157, test/MAPE: 7.9504, test/RMSE: 5.9386]
2025-06-15 21:48:03,834 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_024.pt saved
2025-06-15 21:48:03,834 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:14:37
2025-06-15 21:48:03,834 - easytorch-training - INFO - Epoch 25 / 100
100%|███████████████████████████████████████| 1498/1498 [06:02<00:00,  4.13it/s]
2025-06-15 21:54:06,477 - easytorch-training - INFO - Result <train>: [train/time: 362.64 (s), train/lr: 1.00e-04, train/loss: 2.6294, train/MAE: 2.6294, train/MAPE: 6.9130, train/RMSE: 5.1865]
2025-06-15 21:54:06,478 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:41<00:00,  5.08it/s]
2025-06-15 21:54:48,447 - easytorch-training - INFO - Result <val>: [val/time: 41.97 (s), val/loss: 2.6636, val/MAE: 2.6636, val/MAPE: 7.2984, val/RMSE: 5.0631]
100%|█████████████████████████████████████████| 427/427 [01:23<00:00,  5.12it/s]
2025-06-15 21:56:11,821 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6221, Test MAPE: 6.7679, Test RMSE: 5.0308
2025-06-15 21:56:11,823 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9575, Test MAPE: 8.1304, Test RMSE: 6.0098
2025-06-15 21:56:11,826 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3455, Test MAPE: 9.7948, Test RMSE: 7.0557
2025-06-15 21:56:11,848 - easytorch-training - INFO - Result <test>: [test/time: 83.40 (s), test/loss: 2.7548, test/MAE: 2.9223, test/MAPE: 8.0311, test/RMSE: 5.9673]
2025-06-15 21:56:13,058 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_025.pt saved
2025-06-15 21:56:13,058 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:14:15
2025-06-15 21:56:13,058 - easytorch-training - INFO - Epoch 26 / 100
100%|███████████████████████████████████████| 1498/1498 [06:02<00:00,  4.13it/s]
2025-06-15 22:02:15,346 - easytorch-training - INFO - Result <train>: [train/time: 362.29 (s), train/lr: 1.00e-04, train/loss: 2.6259, train/MAE: 2.6259, train/MAPE: 6.8999, train/RMSE: 5.1796]
2025-06-15 22:02:15,347 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:41<00:00,  5.08it/s]
2025-06-15 22:02:57,313 - easytorch-training - INFO - Result <val>: [val/time: 41.97 (s), val/loss: 2.6627, val/MAE: 2.6627, val/MAPE: 7.2794, val/RMSE: 5.0606]
100%|█████████████████████████████████████████| 427/427 [01:23<00:00,  5.12it/s]
2025-06-15 22:04:20,730 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6227, Test MAPE: 6.7945, Test RMSE: 5.0232
2025-06-15 22:04:20,732 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9619, Test MAPE: 8.1308, Test RMSE: 6.0122
2025-06-15 22:04:20,734 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3434, Test MAPE: 9.7111, Test RMSE: 7.0377
2025-06-15 22:04:20,755 - easytorch-training - INFO - Result <test>: [test/time: 83.44 (s), test/loss: 2.7554, test/MAE: 2.9241, test/MAPE: 8.0190, test/RMSE: 5.9634]
2025-06-15 22:04:21,945 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_026.pt saved
2025-06-15 22:04:21,946 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:13:54
2025-06-15 22:04:21,946 - easytorch-training - INFO - Epoch 27 / 100
100%|███████████████████████████████████████| 1498/1498 [06:03<00:00,  4.13it/s]
2025-06-15 22:10:24,953 - easytorch-training - INFO - Result <train>: [train/time: 363.01 (s), train/lr: 1.00e-04, train/loss: 2.6224, train/MAE: 2.6224, train/MAPE: 6.8843, train/RMSE: 5.1674]
2025-06-15 22:10:24,953 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:42<00:00,  5.07it/s]
2025-06-15 22:11:07,007 - easytorch-training - INFO - Result <val>: [val/time: 42.05 (s), val/loss: 2.6665, val/MAE: 2.6665, val/MAPE: 7.2869, val/RMSE: 5.0719]
100%|█████████████████████████████████████████| 427/427 [01:23<00:00,  5.12it/s]
2025-06-15 22:12:30,471 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6252, Test MAPE: 6.8219, Test RMSE: 5.0370
2025-06-15 22:12:30,473 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9682, Test MAPE: 8.1782, Test RMSE: 6.0345
2025-06-15 22:12:30,475 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3559, Test MAPE: 9.7697, Test RMSE: 7.0763
2025-06-15 22:12:30,500 - easytorch-training - INFO - Result <test>: [test/time: 83.49 (s), test/loss: 2.7607, test/MAE: 2.9305, test/MAPE: 8.0579, test/RMSE: 5.9872]
2025-06-15 22:12:31,746 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_027.pt saved
2025-06-15 22:12:31,747 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:13:38
2025-06-15 22:12:31,747 - easytorch-training - INFO - Epoch 28 / 100
100%|███████████████████████████████████████| 1498/1498 [06:02<00:00,  4.13it/s]
2025-06-15 22:18:34,049 - easytorch-training - INFO - Result <train>: [train/time: 362.30 (s), train/lr: 1.00e-04, train/loss: 2.6186, train/MAE: 2.6186, train/MAPE: 6.8691, train/RMSE: 5.1595]
2025-06-15 22:18:34,050 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:42<00:00,  5.06it/s]
2025-06-15 22:19:16,120 - easytorch-training - INFO - Result <val>: [val/time: 42.07 (s), val/loss: 2.6714, val/MAE: 2.6714, val/MAPE: 7.3079, val/RMSE: 5.0824]
100%|█████████████████████████████████████████| 427/427 [01:23<00:00,  5.11it/s]
2025-06-15 22:20:39,754 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6273, Test MAPE: 6.7998, Test RMSE: 5.0409
2025-06-15 22:20:39,756 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9685, Test MAPE: 8.1443, Test RMSE: 6.0322
2025-06-15 22:20:39,758 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3560, Test MAPE: 9.7318, Test RMSE: 7.0781
2025-06-15 22:20:39,781 - easytorch-training - INFO - Result <test>: [test/time: 83.66 (s), test/loss: 2.7639, test/MAE: 2.9317, test/MAPE: 8.0313, test/RMSE: 5.9893]
2025-06-15 22:20:41,037 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_028.pt saved
2025-06-15 22:20:41,038 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:13:21
2025-06-15 22:20:41,038 - easytorch-training - INFO - Epoch 29 / 100
100%|███████████████████████████████████████| 1498/1498 [06:02<00:00,  4.13it/s]
2025-06-15 22:26:43,437 - easytorch-training - INFO - Result <train>: [train/time: 362.40 (s), train/lr: 1.00e-04, train/loss: 2.6172, train/MAE: 2.6172, train/MAPE: 6.8655, train/RMSE: 5.1550]
2025-06-15 22:26:43,459 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:41<00:00,  5.08it/s]
2025-06-15 22:27:25,385 - easytorch-training - INFO - Result <val>: [val/time: 41.93 (s), val/loss: 2.6769, val/MAE: 2.6769, val/MAPE: 7.2437, val/RMSE: 5.0846]
100%|█████████████████████████████████████████| 427/427 [01:23<00:00,  5.12it/s]
2025-06-15 22:28:48,812 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6273, Test MAPE: 6.7341, Test RMSE: 5.0360
2025-06-15 22:28:48,815 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9686, Test MAPE: 8.0585, Test RMSE: 6.0244
2025-06-15 22:28:48,817 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3530, Test MAPE: 9.6865, Test RMSE: 7.0560
2025-06-15 22:28:48,839 - easytorch-training - INFO - Result <test>: [test/time: 83.45 (s), test/loss: 2.7621, test/MAE: 2.9318, test/MAPE: 7.9606, test/RMSE: 5.9777]
2025-06-15 22:28:50,042 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_029.pt saved
2025-06-15 22:28:50,042 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:13:04
2025-06-15 22:28:50,042 - easytorch-training - INFO - Epoch 30 / 100
100%|███████████████████████████████████████| 1498/1498 [06:01<00:00,  4.14it/s]
2025-06-15 22:34:51,755 - easytorch-training - INFO - Result <train>: [train/time: 361.71 (s), train/lr: 1.00e-04, train/loss: 2.6153, train/MAE: 2.6153, train/MAPE: 6.8573, train/RMSE: 5.1481]
2025-06-15 22:34:51,756 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:41<00:00,  5.11it/s]
2025-06-15 22:35:33,439 - easytorch-training - INFO - Result <val>: [val/time: 41.68 (s), val/loss: 2.6715, val/MAE: 2.6715, val/MAPE: 7.4307, val/RMSE: 5.0993]
100%|█████████████████████████████████████████| 427/427 [01:22<00:00,  5.15it/s]
2025-06-15 22:36:56,395 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6320, Test MAPE: 6.9061, Test RMSE: 5.0688
2025-06-15 22:36:56,397 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9788, Test MAPE: 8.3167, Test RMSE: 6.0905
2025-06-15 22:36:56,399 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3632, Test MAPE: 9.9258, Test RMSE: 7.1303
2025-06-15 22:36:56,421 - easytorch-training - INFO - Result <test>: [test/time: 82.98 (s), test/loss: 2.7705, test/MAE: 2.9384, test/MAPE: 8.1819, test/RMSE: 6.0350]
2025-06-15 22:36:57,631 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_030.pt saved
2025-06-15 22:36:58,967 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:12:48
2025-06-15 22:36:58,967 - easytorch-training - INFO - Epoch 31 / 100
100%|███████████████████████████████████████| 1498/1498 [06:00<00:00,  4.16it/s]
2025-06-15 22:42:59,079 - easytorch-training - INFO - Result <train>: [train/time: 360.11 (s), train/lr: 1.00e-05, train/loss: 2.6027, train/MAE: 2.6027, train/MAPE: 6.8161, train/RMSE: 5.1200]
2025-06-15 22:42:59,080 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:41<00:00,  5.12it/s]
2025-06-15 22:43:40,721 - easytorch-training - INFO - Result <val>: [val/time: 41.64 (s), val/loss: 2.6682, val/MAE: 2.6682, val/MAPE: 7.3147, val/RMSE: 5.0818]
100%|█████████████████████████████████████████| 427/427 [01:22<00:00,  5.15it/s]
2025-06-15 22:45:03,621 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6231, Test MAPE: 6.7911, Test RMSE: 5.0330
2025-06-15 22:45:03,623 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9670, Test MAPE: 8.1544, Test RMSE: 6.0366
2025-06-15 22:45:03,626 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3531, Test MAPE: 9.7813, Test RMSE: 7.0772
2025-06-15 22:45:03,648 - easytorch-training - INFO - Result <test>: [test/time: 82.93 (s), test/loss: 2.7602, test/MAE: 2.9281, test/MAPE: 8.0432, test/RMSE: 5.9862]
2025-06-15 22:45:04,790 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_031.pt saved
2025-06-15 22:45:04,790 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:12:23
2025-06-15 22:45:04,790 - easytorch-training - INFO - Epoch 32 / 100
100%|███████████████████████████████████████| 1498/1498 [05:59<00:00,  4.16it/s]
2025-06-15 22:51:04,693 - easytorch-training - INFO - Result <train>: [train/time: 359.90 (s), train/lr: 1.00e-05, train/loss: 2.5998, train/MAE: 2.5998, train/MAPE: 6.8060, train/RMSE: 5.1139]
2025-06-15 22:51:04,693 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:41<00:00,  5.12it/s]
2025-06-15 22:51:46,303 - easytorch-training - INFO - Result <val>: [val/time: 41.61 (s), val/loss: 2.6703, val/MAE: 2.6703, val/MAPE: 7.3039, val/RMSE: 5.0884]
100%|█████████████████████████████████████████| 427/427 [01:22<00:00,  5.15it/s]
2025-06-15 22:53:09,155 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6253, Test MAPE: 6.7853, Test RMSE: 5.0400
2025-06-15 22:53:09,157 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9695, Test MAPE: 8.1486, Test RMSE: 6.0442
2025-06-15 22:53:09,159 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3558, Test MAPE: 9.7773, Test RMSE: 7.0860
2025-06-15 22:53:09,181 - easytorch-training - INFO - Result <test>: [test/time: 82.88 (s), test/loss: 2.7622, test/MAE: 2.9304, test/MAPE: 8.0377, test/RMSE: 5.9937]
2025-06-15 22:53:10,306 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_032.pt saved
2025-06-15 22:53:10,306 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:11:58
2025-06-15 22:53:10,307 - easytorch-training - INFO - Epoch 33 / 100
100%|███████████████████████████████████████| 1498/1498 [05:59<00:00,  4.16it/s]
2025-06-15 22:59:10,169 - easytorch-training - INFO - Result <train>: [train/time: 359.86 (s), train/lr: 1.00e-05, train/loss: 2.5999, train/MAE: 2.5999, train/MAPE: 6.8055, train/RMSE: 5.1125]
2025-06-15 22:59:10,170 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:41<00:00,  5.12it/s]
2025-06-15 22:59:51,746 - easytorch-training - INFO - Result <val>: [val/time: 41.58 (s), val/loss: 2.6692, val/MAE: 2.6692, val/MAPE: 7.3232, val/RMSE: 5.0831]
100%|█████████████████████████████████████████| 427/427 [01:22<00:00,  5.16it/s]
2025-06-15 23:01:14,561 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6229, Test MAPE: 6.8021, Test RMSE: 5.0364
2025-06-15 23:01:14,563 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9645, Test MAPE: 8.1629, Test RMSE: 6.0326
2025-06-15 23:01:14,565 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3507, Test MAPE: 9.7919, Test RMSE: 7.0698
2025-06-15 23:01:14,587 - easytorch-training - INFO - Result <test>: [test/time: 82.84 (s), test/loss: 2.7582, test/MAE: 2.9265, test/MAPE: 8.0516, test/RMSE: 5.9835]
2025-06-15 23:01:15,719 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_033.pt saved
2025-06-15 23:01:15,719 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:11:35
2025-06-15 23:01:15,719 - easytorch-training - INFO - Epoch 34 / 100
100%|███████████████████████████████████████| 1498/1498 [06:00<00:00,  4.16it/s]
2025-06-15 23:07:15,804 - easytorch-training - INFO - Result <train>: [train/time: 360.08 (s), train/lr: 1.00e-05, train/loss: 2.5990, train/MAE: 2.5990, train/MAPE: 6.8029, train/RMSE: 5.1100]
2025-06-15 23:07:15,805 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:41<00:00,  5.12it/s]
2025-06-15 23:07:57,432 - easytorch-training - INFO - Result <val>: [val/time: 41.63 (s), val/loss: 2.6703, val/MAE: 2.6703, val/MAPE: 7.3089, val/RMSE: 5.0874]
100%|█████████████████████████████████████████| 427/427 [01:22<00:00,  5.15it/s]
2025-06-15 23:09:20,273 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6234, Test MAPE: 6.7952, Test RMSE: 5.0373
2025-06-15 23:09:20,275 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9660, Test MAPE: 8.1517, Test RMSE: 6.0387
2025-06-15 23:09:20,277 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3515, Test MAPE: 9.7629, Test RMSE: 7.0757
2025-06-15 23:09:20,298 - easytorch-training - INFO - Result <test>: [test/time: 82.87 (s), test/loss: 2.7593, test/MAE: 2.9274, test/MAPE: 8.0354, test/RMSE: 5.9877]
2025-06-15 23:09:21,414 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_034.pt saved
2025-06-15 23:09:21,415 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:11:14
2025-06-15 23:09:21,415 - easytorch-training - INFO - Epoch 35 / 100
100%|███████████████████████████████████████| 1498/1498 [06:00<00:00,  4.16it/s]
2025-06-15 23:15:21,535 - easytorch-training - INFO - Result <train>: [train/time: 360.12 (s), train/lr: 1.00e-05, train/loss: 2.5991, train/MAE: 2.5991, train/MAPE: 6.8048, train/RMSE: 5.1098]
2025-06-15 23:15:21,536 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:41<00:00,  5.12it/s]
2025-06-15 23:16:03,154 - easytorch-training - INFO - Result <val>: [val/time: 41.62 (s), val/loss: 2.6703, val/MAE: 2.6703, val/MAPE: 7.3260, val/RMSE: 5.0888]
100%|█████████████████████████████████████████| 427/427 [01:22<00:00,  5.16it/s]
2025-06-15 23:17:25,981 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6253, Test MAPE: 6.8197, Test RMSE: 5.0431
2025-06-15 23:17:25,984 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9687, Test MAPE: 8.1772, Test RMSE: 6.0469
2025-06-15 23:17:25,985 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3549, Test MAPE: 9.7965, Test RMSE: 7.0863
2025-06-15 23:17:26,006 - easytorch-training - INFO - Result <test>: [test/time: 82.85 (s), test/loss: 2.7618, test/MAE: 2.9297, test/MAPE: 8.0635, test/RMSE: 5.9954]
2025-06-15 23:17:27,191 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_035.pt saved
2025-06-15 23:17:27,191 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:10:55
2025-06-15 23:17:27,191 - easytorch-training - INFO - Epoch 36 / 100
100%|███████████████████████████████████████| 1498/1498 [06:01<00:00,  4.15it/s]
2025-06-15 23:23:28,227 - easytorch-training - INFO - Result <train>: [train/time: 361.04 (s), train/lr: 1.00e-05, train/loss: 2.5985, train/MAE: 2.5985, train/MAPE: 6.7996, train/RMSE: 5.1098]
2025-06-15 23:23:28,228 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:41<00:00,  5.12it/s]
2025-06-15 23:24:09,837 - easytorch-training - INFO - Result <val>: [val/time: 41.61 (s), val/loss: 2.6694, val/MAE: 2.6694, val/MAPE: 7.3006, val/RMSE: 5.0856]
100%|█████████████████████████████████████████| 427/427 [01:22<00:00,  5.16it/s]
2025-06-15 23:25:32,673 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6234, Test MAPE: 6.7997, Test RMSE: 5.0383
2025-06-15 23:25:32,675 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9659, Test MAPE: 8.1416, Test RMSE: 6.0360
2025-06-15 23:25:32,677 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3527, Test MAPE: 9.7439, Test RMSE: 7.0754
2025-06-15 23:25:32,699 - easytorch-training - INFO - Result <test>: [test/time: 82.86 (s), test/loss: 2.7599, test/MAE: 2.9277, test/MAPE: 8.0292, test/RMSE: 5.9867]
2025-06-15 23:25:33,907 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_036.pt saved
2025-06-15 23:25:33,907 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:10:39
2025-06-15 23:25:33,907 - easytorch-training - INFO - Epoch 37 / 100
100%|███████████████████████████████████████| 1498/1498 [06:00<00:00,  4.16it/s]
2025-06-15 23:31:34,263 - easytorch-training - INFO - Result <train>: [train/time: 360.36 (s), train/lr: 1.00e-05, train/loss: 2.5978, train/MAE: 2.5978, train/MAPE: 6.7987, train/RMSE: 5.1078]
2025-06-15 23:31:34,264 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:41<00:00,  5.12it/s]
2025-06-15 23:32:15,864 - easytorch-training - INFO - Result <val>: [val/time: 41.60 (s), val/loss: 2.6699, val/MAE: 2.6699, val/MAPE: 7.2985, val/RMSE: 5.0862]
100%|█████████████████████████████████████████| 427/427 [01:22<00:00,  5.15it/s]
2025-06-15 23:33:38,719 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6250, Test MAPE: 6.7940, Test RMSE: 5.0416
2025-06-15 23:33:38,721 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9680, Test MAPE: 8.1514, Test RMSE: 6.0432
2025-06-15 23:33:38,723 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3536, Test MAPE: 9.7644, Test RMSE: 7.0830
2025-06-15 23:33:38,747 - easytorch-training - INFO - Result <test>: [test/time: 82.88 (s), test/loss: 2.7612, test/MAE: 2.9292, test/MAPE: 8.0370, test/RMSE: 5.9927]
2025-06-15 23:33:39,898 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_037.pt saved
2025-06-15 23:33:39,898 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:10:22
2025-06-15 23:33:39,898 - easytorch-training - INFO - Epoch 38 / 100
100%|███████████████████████████████████████| 1498/1498 [06:00<00:00,  4.15it/s]
2025-06-15 23:39:40,440 - easytorch-training - INFO - Result <train>: [train/time: 360.54 (s), train/lr: 1.00e-05, train/loss: 2.5975, train/MAE: 2.5975, train/MAPE: 6.7974, train/RMSE: 5.1057]
2025-06-15 23:39:40,441 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:41<00:00,  5.12it/s]
2025-06-15 23:40:22,083 - easytorch-training - INFO - Result <val>: [val/time: 41.64 (s), val/loss: 2.6696, val/MAE: 2.6696, val/MAPE: 7.2993, val/RMSE: 5.0866]
100%|█████████████████████████████████████████| 427/427 [01:22<00:00,  5.15it/s]
2025-06-15 23:41:44,992 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6239, Test MAPE: 6.7957, Test RMSE: 5.0384
2025-06-15 23:41:44,996 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9665, Test MAPE: 8.1465, Test RMSE: 6.0370
2025-06-15 23:41:44,999 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3535, Test MAPE: 9.7422, Test RMSE: 7.0769
2025-06-15 23:41:45,020 - easytorch-training - INFO - Result <test>: [test/time: 82.94 (s), test/loss: 2.7600, test/MAE: 2.9282, test/MAPE: 8.0286, test/RMSE: 5.9871]
2025-06-15 23:41:46,149 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_038.pt saved
2025-06-15 23:41:46,150 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:10:06
2025-06-15 23:41:46,150 - easytorch-training - INFO - Epoch 39 / 100
100%|███████████████████████████████████████| 1498/1498 [05:59<00:00,  4.16it/s]
2025-06-15 23:47:46,143 - easytorch-training - INFO - Result <train>: [train/time: 359.99 (s), train/lr: 1.00e-05, train/loss: 2.5969, train/MAE: 2.5969, train/MAPE: 6.7975, train/RMSE: 5.1041]
2025-06-15 23:47:46,145 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:41<00:00,  5.12it/s]
2025-06-15 23:48:27,779 - easytorch-training - INFO - Result <val>: [val/time: 41.63 (s), val/loss: 2.6699, val/MAE: 2.6699, val/MAPE: 7.2971, val/RMSE: 5.0844]
100%|█████████████████████████████████████████| 427/427 [01:22<00:00,  5.15it/s]
2025-06-15 23:49:50,638 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6236, Test MAPE: 6.7791, Test RMSE: 5.0363
2025-06-15 23:49:50,640 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9663, Test MAPE: 8.1388, Test RMSE: 6.0354
2025-06-15 23:49:50,642 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3528, Test MAPE: 9.7412, Test RMSE: 7.0747
2025-06-15 23:49:50,665 - easytorch-training - INFO - Result <test>: [test/time: 82.88 (s), test/loss: 2.7596, test/MAE: 2.9278, test/MAPE: 8.0196, test/RMSE: 5.9857]
2025-06-15 23:49:51,781 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_039.pt saved
2025-06-15 23:49:51,782 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:09:50
2025-06-15 23:49:51,782 - easytorch-training - INFO - Epoch 40 / 100
100%|███████████████████████████████████████| 1498/1498 [06:00<00:00,  4.15it/s]
2025-06-15 23:55:52,438 - easytorch-training - INFO - Result <train>: [train/time: 360.66 (s), train/lr: 1.00e-05, train/loss: 2.5968, train/MAE: 2.5968, train/MAPE: 6.7951, train/RMSE: 5.1017]
2025-06-15 23:55:52,439 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:41<00:00,  5.10it/s]
2025-06-15 23:56:34,185 - easytorch-training - INFO - Result <val>: [val/time: 41.75 (s), val/loss: 2.6699, val/MAE: 2.6699, val/MAPE: 7.3147, val/RMSE: 5.0822]
100%|█████████████████████████████████████████| 427/427 [01:23<00:00,  5.14it/s]
2025-06-15 23:57:57,327 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6237, Test MAPE: 6.8006, Test RMSE: 5.0351
2025-06-15 23:57:57,330 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9671, Test MAPE: 8.1641, Test RMSE: 6.0371
2025-06-15 23:57:57,331 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3524, Test MAPE: 9.7697, Test RMSE: 7.0731
2025-06-15 23:57:57,354 - easytorch-training - INFO - Result <test>: [test/time: 83.17 (s), test/loss: 2.7599, test/MAE: 2.9281, test/MAPE: 8.0462, test/RMSE: 5.9863]
2025-06-15 23:57:58,651 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_040.pt saved
2025-06-15 23:57:59,838 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:09:41
2025-06-15 23:57:59,838 - easytorch-training - INFO - Epoch 41 / 100
100%|███████████████████████████████████████| 1498/1498 [06:01<00:00,  4.14it/s]
2025-06-16 00:04:01,427 - easytorch-training - INFO - Result <train>: [train/time: 361.59 (s), train/lr: 1.00e-05, train/loss: 2.5957, train/MAE: 2.5957, train/MAPE: 6.7902, train/RMSE: 5.1006]
2025-06-16 00:04:01,429 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:41<00:00,  5.08it/s]
2025-06-16 00:04:43,338 - easytorch-training - INFO - Result <val>: [val/time: 41.91 (s), val/loss: 2.6716, val/MAE: 2.6716, val/MAPE: 7.3174, val/RMSE: 5.0874]
100%|█████████████████████████████████████████| 427/427 [01:23<00:00,  5.11it/s]
2025-06-16 00:06:06,896 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6250, Test MAPE: 6.8097, Test RMSE: 5.0398
2025-06-16 00:06:06,898 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9692, Test MAPE: 8.1691, Test RMSE: 6.0439
2025-06-16 00:06:06,901 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3557, Test MAPE: 9.7825, Test RMSE: 7.0834
2025-06-16 00:06:06,923 - easytorch-training - INFO - Result <test>: [test/time: 83.58 (s), test/loss: 2.7621, test/MAE: 2.9305, test/MAPE: 8.0534, test/RMSE: 5.9940]
2025-06-16 00:06:08,116 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_041.pt saved
2025-06-16 00:06:08,116 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:09:33
2025-06-16 00:06:08,116 - easytorch-training - INFO - Epoch 42 / 100
100%|███████████████████████████████████████| 1498/1498 [06:02<00:00,  4.13it/s]
2025-06-16 00:12:10,719 - easytorch-training - INFO - Result <train>: [train/time: 362.60 (s), train/lr: 1.00e-05, train/loss: 2.5958, train/MAE: 2.5958, train/MAPE: 6.7896, train/RMSE: 5.1023]
2025-06-16 00:12:10,721 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:41<00:00,  5.07it/s]
2025-06-16 00:12:52,719 - easytorch-training - INFO - Result <val>: [val/time: 42.00 (s), val/loss: 2.6715, val/MAE: 2.6715, val/MAPE: 7.3039, val/RMSE: 5.0888]
100%|█████████████████████████████████████████| 427/427 [01:23<00:00,  5.11it/s]
2025-06-16 00:14:16,293 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6239, Test MAPE: 6.7948, Test RMSE: 5.0382
2025-06-16 00:14:16,295 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9680, Test MAPE: 8.1523, Test RMSE: 6.0415
2025-06-16 00:14:16,297 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3550, Test MAPE: 9.7590, Test RMSE: 7.0806
2025-06-16 00:14:16,320 - easytorch-training - INFO - Result <test>: [test/time: 83.60 (s), test/loss: 2.7605, test/MAE: 2.9291, test/MAPE: 8.0345, test/RMSE: 5.9900]
2025-06-16 00:14:17,571 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_042.pt saved
2025-06-16 00:14:17,571 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:09:27
2025-06-16 00:14:17,571 - easytorch-training - INFO - Epoch 43 / 100
100%|███████████████████████████████████████| 1498/1498 [06:02<00:00,  4.13it/s]
2025-06-16 00:20:20,510 - easytorch-training - INFO - Result <train>: [train/time: 362.94 (s), train/lr: 1.00e-05, train/loss: 2.5958, train/MAE: 2.5958, train/MAPE: 6.7914, train/RMSE: 5.1017]
2025-06-16 00:20:20,511 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:42<00:00,  5.07it/s]
2025-06-16 00:21:02,553 - easytorch-training - INFO - Result <val>: [val/time: 42.04 (s), val/loss: 2.6729, val/MAE: 2.6729, val/MAPE: 7.2860, val/RMSE: 5.0921]
100%|█████████████████████████████████████████| 427/427 [01:23<00:00,  5.10it/s]
2025-06-16 00:22:26,215 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6246, Test MAPE: 6.7750, Test RMSE: 5.0376
2025-06-16 00:22:26,217 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9705, Test MAPE: 8.1293, Test RMSE: 6.0459
2025-06-16 00:22:26,219 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3565, Test MAPE: 9.7492, Test RMSE: 7.0854
2025-06-16 00:22:26,244 - easytorch-training - INFO - Result <test>: [test/time: 83.69 (s), test/loss: 2.7622, test/MAE: 2.9312, test/MAPE: 8.0163, test/RMSE: 5.9945]
2025-06-16 00:22:27,599 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_043.pt saved
2025-06-16 00:22:27,599 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:09:24
2025-06-16 00:22:27,599 - easytorch-training - INFO - Epoch 44 / 100
100%|███████████████████████████████████████| 1498/1498 [06:04<00:00,  4.11it/s]
2025-06-16 00:28:31,914 - easytorch-training - INFO - Result <train>: [train/time: 364.31 (s), train/lr: 1.00e-05, train/loss: 2.5954, train/MAE: 2.5954, train/MAPE: 6.7905, train/RMSE: 5.1003]
2025-06-16 00:28:31,915 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:42<00:00,  5.02it/s]
2025-06-16 00:29:14,380 - easytorch-training - INFO - Result <val>: [val/time: 42.46 (s), val/loss: 2.6716, val/MAE: 2.6716, val/MAPE: 7.3067, val/RMSE: 5.0885]
100%|█████████████████████████████████████████| 427/427 [01:23<00:00,  5.08it/s]
2025-06-16 00:30:38,380 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6259, Test MAPE: 6.7941, Test RMSE: 5.0408
2025-06-16 00:30:38,382 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9709, Test MAPE: 8.1631, Test RMSE: 6.0474
2025-06-16 00:30:38,384 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3576, Test MAPE: 9.7789, Test RMSE: 7.0867
2025-06-16 00:30:38,405 - easytorch-training - INFO - Result <test>: [test/time: 84.02 (s), test/loss: 2.7630, test/MAE: 2.9317, test/MAPE: 8.0446, test/RMSE: 5.9956]
2025-06-16 00:30:39,610 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_044.pt saved
2025-06-16 00:30:39,610 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:09:25
2025-06-16 00:30:39,610 - easytorch-training - INFO - Epoch 45 / 100
100%|███████████████████████████████████████| 1498/1498 [06:05<00:00,  4.09it/s]
2025-06-16 00:36:45,582 - easytorch-training - INFO - Result <train>: [train/time: 365.97 (s), train/lr: 1.00e-05, train/loss: 2.5951, train/MAE: 2.5951, train/MAPE: 6.7871, train/RMSE: 5.0984]
2025-06-16 00:36:45,583 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:42<00:00,  5.03it/s]
2025-06-16 00:37:27,960 - easytorch-training - INFO - Result <val>: [val/time: 42.38 (s), val/loss: 2.6704, val/MAE: 2.6704, val/MAPE: 7.3155, val/RMSE: 5.0852]
100%|█████████████████████████████████████████| 427/427 [01:24<00:00,  5.06it/s]
2025-06-16 00:38:52,381 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6243, Test MAPE: 6.7952, Test RMSE: 5.0370
2025-06-16 00:38:52,384 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9682, Test MAPE: 8.1638, Test RMSE: 6.0401
2025-06-16 00:38:52,386 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3548, Test MAPE: 9.7945, Test RMSE: 7.0822
2025-06-16 00:38:52,413 - easytorch-training - INFO - Result <test>: [test/time: 84.45 (s), test/loss: 2.7609, test/MAE: 2.9294, test/MAPE: 8.0508, test/RMSE: 5.9903]
2025-06-16 00:38:53,677 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_045.pt saved
2025-06-16 00:38:53,677 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:09:31
2025-06-16 00:38:53,677 - easytorch-training - INFO - Epoch 46 / 100
100%|███████████████████████████████████████| 1498/1498 [06:06<00:00,  4.09it/s]
2025-06-16 00:44:59,968 - easytorch-training - INFO - Result <train>: [train/time: 366.29 (s), train/lr: 1.00e-05, train/loss: 2.5942, train/MAE: 2.5942, train/MAPE: 6.7861, train/RMSE: 5.0970]
2025-06-16 00:44:59,969 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:42<00:00,  5.02it/s]
2025-06-16 00:45:42,362 - easytorch-training - INFO - Result <val>: [val/time: 42.39 (s), val/loss: 2.6694, val/MAE: 2.6694, val/MAPE: 7.2807, val/RMSE: 5.0834]
100%|█████████████████████████████████████████| 427/427 [01:24<00:00,  5.06it/s]
2025-06-16 00:47:06,776 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6230, Test MAPE: 6.7783, Test RMSE: 5.0340
2025-06-16 00:47:06,779 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9675, Test MAPE: 8.1273, Test RMSE: 6.0379
2025-06-16 00:47:06,781 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3524, Test MAPE: 9.7294, Test RMSE: 7.0731
2025-06-16 00:47:06,805 - easytorch-training - INFO - Result <test>: [test/time: 84.44 (s), test/loss: 2.7592, test/MAE: 2.9279, test/MAPE: 8.0110, test/RMSE: 5.9854]
2025-06-16 00:47:08,034 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_046.pt saved
2025-06-16 00:47:08,034 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:09:37
2025-06-16 00:47:08,034 - easytorch-training - INFO - Epoch 47 / 100
100%|███████████████████████████████████████| 1498/1498 [06:06<00:00,  4.09it/s]
2025-06-16 00:53:14,333 - easytorch-training - INFO - Result <train>: [train/time: 366.30 (s), train/lr: 1.00e-05, train/loss: 2.5932, train/MAE: 2.5932, train/MAPE: 6.7817, train/RMSE: 5.0938]
2025-06-16 00:53:14,334 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:42<00:00,  5.01it/s]
2025-06-16 00:53:56,819 - easytorch-training - INFO - Result <val>: [val/time: 42.48 (s), val/loss: 2.6712, val/MAE: 2.6712, val/MAPE: 7.3446, val/RMSE: 5.0897]
100%|█████████████████████████████████████████| 427/427 [01:24<00:00,  5.07it/s]
2025-06-16 00:55:21,068 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6261, Test MAPE: 6.8277, Test RMSE: 5.0442
2025-06-16 00:55:21,070 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9710, Test MAPE: 8.2000, Test RMSE: 6.0520
2025-06-16 00:55:21,072 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3580, Test MAPE: 9.8459, Test RMSE: 7.0905
2025-06-16 00:55:21,094 - easytorch-training - INFO - Result <test>: [test/time: 84.27 (s), test/loss: 2.7633, test/MAE: 2.9319, test/MAPE: 8.0878, test/RMSE: 6.0001]
2025-06-16 00:55:22,335 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_047.pt saved
2025-06-16 00:55:22,335 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:09:42
2025-06-16 00:55:22,335 - easytorch-training - INFO - Epoch 48 / 100
100%|███████████████████████████████████████| 1498/1498 [06:06<00:00,  4.09it/s]
2025-06-16 01:01:28,343 - easytorch-training - INFO - Result <train>: [train/time: 366.01 (s), train/lr: 1.00e-05, train/loss: 2.5940, train/MAE: 2.5940, train/MAPE: 6.7847, train/RMSE: 5.0979]
2025-06-16 01:01:28,357 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:42<00:00,  5.02it/s]
2025-06-16 01:02:10,790 - easytorch-training - INFO - Result <val>: [val/time: 42.43 (s), val/loss: 2.6719, val/MAE: 2.6719, val/MAPE: 7.3034, val/RMSE: 5.0936]
100%|█████████████████████████████████████████| 427/427 [01:24<00:00,  5.06it/s]
2025-06-16 01:03:35,101 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6260, Test MAPE: 6.7882, Test RMSE: 5.0448
2025-06-16 01:03:35,103 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9707, Test MAPE: 8.1495, Test RMSE: 6.0499
2025-06-16 01:03:35,106 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3568, Test MAPE: 9.7695, Test RMSE: 7.0898
2025-06-16 01:03:35,131 - easytorch-training - INFO - Result <test>: [test/time: 84.34 (s), test/loss: 2.7628, test/MAE: 2.9316, test/MAPE: 8.0367, test/RMSE: 5.9989]
2025-06-16 01:03:36,350 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_048.pt saved
2025-06-16 01:03:36,350 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:09:47
2025-06-16 01:03:36,350 - easytorch-training - INFO - Epoch 49 / 100
100%|███████████████████████████████████████| 1498/1498 [06:06<00:00,  4.09it/s]
2025-06-16 01:09:42,761 - easytorch-training - INFO - Result <train>: [train/time: 366.41 (s), train/lr: 1.00e-05, train/loss: 2.5929, train/MAE: 2.5929, train/MAPE: 6.7807, train/RMSE: 5.0951]
2025-06-16 01:09:42,762 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:42<00:00,  5.03it/s]
2025-06-16 01:10:25,116 - easytorch-training - INFO - Result <val>: [val/time: 42.35 (s), val/loss: 2.6716, val/MAE: 2.6716, val/MAPE: 7.3160, val/RMSE: 5.0905]
100%|█████████████████████████████████████████| 427/427 [01:24<00:00,  5.06it/s]
2025-06-16 01:11:49,494 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6251, Test MAPE: 6.8064, Test RMSE: 5.0422
2025-06-16 01:11:49,497 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9700, Test MAPE: 8.1735, Test RMSE: 6.0479
2025-06-16 01:11:49,499 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3589, Test MAPE: 9.7911, Test RMSE: 7.0946
2025-06-16 01:11:49,523 - easytorch-training - INFO - Result <test>: [test/time: 84.41 (s), test/loss: 2.7630, test/MAE: 2.9316, test/MAPE: 8.0539, test/RMSE: 5.9990]
2025-06-16 01:11:50,793 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_049.pt saved
2025-06-16 01:11:50,793 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:09:53
2025-06-16 01:11:50,793 - easytorch-training - INFO - Epoch 50 / 100
100%|███████████████████████████████████████| 1498/1498 [06:06<00:00,  4.09it/s]
2025-06-16 01:17:57,081 - easytorch-training - INFO - Result <train>: [train/time: 366.29 (s), train/lr: 1.00e-05, train/loss: 2.5928, train/MAE: 2.5928, train/MAPE: 6.7795, train/RMSE: 5.0931]
2025-06-16 01:17:57,082 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:42<00:00,  5.00it/s]
2025-06-16 01:18:39,657 - easytorch-training - INFO - Result <val>: [val/time: 42.57 (s), val/loss: 2.6730, val/MAE: 2.6730, val/MAPE: 7.3192, val/RMSE: 5.0936]
100%|█████████████████████████████████████████| 427/427 [01:24<00:00,  5.05it/s]
2025-06-16 01:20:04,153 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6256, Test MAPE: 6.7948, Test RMSE: 5.0427
2025-06-16 01:20:04,156 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9707, Test MAPE: 8.1625, Test RMSE: 6.0496
2025-06-16 01:20:04,158 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3585, Test MAPE: 9.7952, Test RMSE: 7.0931
2025-06-16 01:20:04,183 - easytorch-training - INFO - Result <test>: [test/time: 84.52 (s), test/loss: 2.7634, test/MAE: 2.9318, test/MAPE: 8.0490, test/RMSE: 5.9992]
2025-06-16 01:20:05,485 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_050.pt saved
2025-06-16 01:20:06,473 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:10:00
2025-06-16 01:20:06,473 - easytorch-training - INFO - Epoch 51 / 100
100%|███████████████████████████████████████| 1498/1498 [06:05<00:00,  4.10it/s]
2025-06-16 01:26:12,240 - easytorch-training - INFO - Result <train>: [train/time: 365.77 (s), train/lr: 1.00e-05, train/loss: 2.5933, train/MAE: 2.5933, train/MAPE: 6.7824, train/RMSE: 5.0946]
2025-06-16 01:26:12,241 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:42<00:00,  5.04it/s]
2025-06-16 01:26:54,504 - easytorch-training - INFO - Result <val>: [val/time: 42.26 (s), val/loss: 2.6714, val/MAE: 2.6714, val/MAPE: 7.3117, val/RMSE: 5.0872]
100%|█████████████████████████████████████████| 427/427 [01:24<00:00,  5.05it/s]
2025-06-16 01:28:19,092 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6258, Test MAPE: 6.8071, Test RMSE: 5.0414
2025-06-16 01:28:19,094 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9717, Test MAPE: 8.1785, Test RMSE: 6.0507
2025-06-16 01:28:19,096 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3609, Test MAPE: 9.7941, Test RMSE: 7.0993
2025-06-16 01:28:19,121 - easytorch-training - INFO - Result <test>: [test/time: 84.62 (s), test/loss: 2.7647, test/MAE: 2.9327, test/MAPE: 8.0565, test/RMSE: 6.0010]
2025-06-16 01:28:20,426 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_051.pt saved
2025-06-16 01:28:20,426 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:10:04
2025-06-16 01:28:20,426 - easytorch-training - INFO - Epoch 52 / 100
100%|███████████████████████████████████████| 1498/1498 [06:07<00:00,  4.07it/s]
2025-06-16 01:34:28,104 - easytorch-training - INFO - Result <train>: [train/time: 367.68 (s), train/lr: 1.00e-05, train/loss: 2.5920, train/MAE: 2.5920, train/MAPE: 6.7763, train/RMSE: 5.0917]
2025-06-16 01:34:28,105 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:42<00:00,  5.00it/s]
2025-06-16 01:35:10,744 - easytorch-training - INFO - Result <val>: [val/time: 42.64 (s), val/loss: 2.6708, val/MAE: 2.6708, val/MAPE: 7.2814, val/RMSE: 5.0908]
100%|█████████████████████████████████████████| 427/427 [01:24<00:00,  5.03it/s]
2025-06-16 01:36:35,665 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6241, Test MAPE: 6.7799, Test RMSE: 5.0401
2025-06-16 01:36:35,667 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9690, Test MAPE: 8.1361, Test RMSE: 6.0464
2025-06-16 01:36:35,669 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3563, Test MAPE: 9.7420, Test RMSE: 7.0897
2025-06-16 01:36:35,695 - easytorch-training - INFO - Result <test>: [test/time: 84.95 (s), test/loss: 2.7619, test/MAE: 2.9302, test/MAPE: 8.0181, test/RMSE: 5.9961]
2025-06-16 01:36:37,037 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_052.pt saved
2025-06-16 01:36:37,037 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:10:13
2025-06-16 01:36:37,038 - easytorch-training - INFO - Epoch 53 / 100
100%|███████████████████████████████████████| 1498/1498 [06:07<00:00,  4.07it/s]
2025-06-16 01:42:44,727 - easytorch-training - INFO - Result <train>: [train/time: 367.69 (s), train/lr: 1.00e-05, train/loss: 2.5912, train/MAE: 2.5912, train/MAPE: 6.7727, train/RMSE: 5.0889]
2025-06-16 01:42:44,728 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:42<00:00,  4.99it/s]
2025-06-16 01:43:27,383 - easytorch-training - INFO - Result <val>: [val/time: 42.65 (s), val/loss: 2.6719, val/MAE: 2.6719, val/MAPE: 7.3246, val/RMSE: 5.0913]
100%|█████████████████████████████████████████| 427/427 [01:24<00:00,  5.03it/s]
2025-06-16 01:44:52,357 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6257, Test MAPE: 6.8193, Test RMSE: 5.0449
2025-06-16 01:44:52,358 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9701, Test MAPE: 8.1832, Test RMSE: 6.0494
2025-06-16 01:44:52,360 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3562, Test MAPE: 9.7807, Test RMSE: 7.0855
2025-06-16 01:44:52,382 - easytorch-training - INFO - Result <test>: [test/time: 85.00 (s), test/loss: 2.7623, test/MAE: 2.9308, test/MAPE: 8.0587, test/RMSE: 5.9969]
2025-06-16 01:44:53,633 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_053.pt saved
2025-06-16 01:44:53,633 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:10:22
2025-06-16 01:44:53,633 - easytorch-training - INFO - Epoch 54 / 100
100%|███████████████████████████████████████| 1498/1498 [06:07<00:00,  4.08it/s]
2025-06-16 01:51:01,236 - easytorch-training - INFO - Result <train>: [train/time: 367.60 (s), train/lr: 1.00e-05, train/loss: 2.5912, train/MAE: 2.5912, train/MAPE: 6.7737, train/RMSE: 5.0900]
2025-06-16 01:51:01,244 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:42<00:00,  5.00it/s]
2025-06-16 01:51:43,888 - easytorch-training - INFO - Result <val>: [val/time: 42.64 (s), val/loss: 2.6717, val/MAE: 2.6717, val/MAPE: 7.3023, val/RMSE: 5.0909]
100%|█████████████████████████████████████████| 427/427 [01:24<00:00,  5.03it/s]
2025-06-16 01:53:08,810 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6236, Test MAPE: 6.7877, Test RMSE: 5.0365
2025-06-16 01:53:08,812 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9689, Test MAPE: 8.1534, Test RMSE: 6.0422
2025-06-16 01:53:08,814 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3568, Test MAPE: 9.7535, Test RMSE: 7.0849
2025-06-16 01:53:08,836 - easytorch-training - INFO - Result <test>: [test/time: 84.95 (s), test/loss: 2.7613, test/MAE: 2.9300, test/MAPE: 8.0316, test/RMSE: 5.9919]
2025-06-16 01:53:10,155 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_054.pt saved
2025-06-16 01:53:10,156 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:10:30
2025-06-16 01:53:10,156 - easytorch-training - INFO - Epoch 55 / 100
100%|███████████████████████████████████████| 1498/1498 [06:07<00:00,  4.07it/s]
2025-06-16 01:59:18,031 - easytorch-training - INFO - Result <train>: [train/time: 367.88 (s), train/lr: 1.00e-05, train/loss: 2.5904, train/MAE: 2.5904, train/MAPE: 6.7688, train/RMSE: 5.0874]
2025-06-16 01:59:18,033 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:42<00:00,  4.99it/s]
2025-06-16 02:00:00,736 - easytorch-training - INFO - Result <val>: [val/time: 42.70 (s), val/loss: 2.6720, val/MAE: 2.6720, val/MAPE: 7.3099, val/RMSE: 5.0896]
100%|█████████████████████████████████████████| 427/427 [01:24<00:00,  5.04it/s]
2025-06-16 02:01:25,512 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6250, Test MAPE: 6.7953, Test RMSE: 5.0397
2025-06-16 02:01:25,514 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9703, Test MAPE: 8.1661, Test RMSE: 6.0479
2025-06-16 02:01:25,516 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3587, Test MAPE: 9.7732, Test RMSE: 7.0936
2025-06-16 02:01:25,542 - easytorch-training - INFO - Result <test>: [test/time: 84.80 (s), test/loss: 2.7632, test/MAE: 2.9314, test/MAPE: 8.0448, test/RMSE: 5.9974]
2025-06-16 02:01:26,820 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_055.pt saved
2025-06-16 02:01:26,820 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:10:38
2025-06-16 02:01:26,820 - easytorch-training - INFO - Epoch 56 / 100
100%|███████████████████████████████████████| 1498/1498 [06:07<00:00,  4.07it/s]
2025-06-16 02:07:34,719 - easytorch-training - INFO - Result <train>: [train/time: 367.90 (s), train/lr: 1.00e-05, train/loss: 2.5903, train/MAE: 2.5903, train/MAPE: 6.7720, train/RMSE: 5.0872]
2025-06-16 02:07:34,721 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:42<00:00,  5.00it/s]
2025-06-16 02:08:17,358 - easytorch-training - INFO - Result <val>: [val/time: 42.64 (s), val/loss: 2.6720, val/MAE: 2.6720, val/MAPE: 7.3165, val/RMSE: 5.0942]
100%|█████████████████████████████████████████| 427/427 [01:24<00:00,  5.04it/s]
2025-06-16 02:09:42,043 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6251, Test MAPE: 6.8145, Test RMSE: 5.0434
2025-06-16 02:09:42,045 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9705, Test MAPE: 8.1732, Test RMSE: 6.0524
2025-06-16 02:09:42,048 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3574, Test MAPE: 9.7763, Test RMSE: 7.0929
2025-06-16 02:09:42,071 - easytorch-training - INFO - Result <test>: [test/time: 84.71 (s), test/loss: 2.7625, test/MAE: 2.9312, test/MAPE: 8.0525, test/RMSE: 6.0000]
2025-06-16 02:09:43,362 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_056.pt saved
2025-06-16 02:09:43,363 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:10:46
2025-06-16 02:09:43,363 - easytorch-training - INFO - Epoch 57 / 100
100%|███████████████████████████████████████| 1498/1498 [06:07<00:00,  4.08it/s]
2025-06-16 02:15:50,848 - easytorch-training - INFO - Result <train>: [train/time: 367.48 (s), train/lr: 1.00e-05, train/loss: 2.5911, train/MAE: 2.5911, train/MAPE: 6.7737, train/RMSE: 5.0887]
2025-06-16 02:15:50,849 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:42<00:00,  5.00it/s]
2025-06-16 02:16:33,483 - easytorch-training - INFO - Result <val>: [val/time: 42.63 (s), val/loss: 2.6718, val/MAE: 2.6718, val/MAPE: 7.2906, val/RMSE: 5.0930]
100%|█████████████████████████████████████████| 427/427 [01:24<00:00,  5.04it/s]
2025-06-16 02:17:58,261 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6267, Test MAPE: 6.7904, Test RMSE: 5.0454
2025-06-16 02:17:58,263 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9734, Test MAPE: 8.1499, Test RMSE: 6.0567
2025-06-16 02:17:58,266 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3605, Test MAPE: 9.7429, Test RMSE: 7.0966
2025-06-16 02:17:58,291 - easytorch-training - INFO - Result <test>: [test/time: 84.81 (s), test/loss: 2.7651, test/MAE: 2.9338, test/MAPE: 8.0266, test/RMSE: 6.0041]
2025-06-16 02:17:59,626 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_057.pt saved
2025-06-16 02:17:59,627 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:10:53
2025-06-16 02:17:59,627 - easytorch-training - INFO - Epoch 58 / 100
100%|███████████████████████████████████████| 1498/1498 [06:08<00:00,  4.06it/s]
2025-06-16 02:24:08,162 - easytorch-training - INFO - Result <train>: [train/time: 368.54 (s), train/lr: 1.00e-05, train/loss: 2.5902, train/MAE: 2.5902, train/MAPE: 6.7701, train/RMSE: 5.0850]
2025-06-16 02:24:08,163 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:42<00:00,  5.00it/s]
2025-06-16 02:24:50,776 - easytorch-training - INFO - Result <val>: [val/time: 42.61 (s), val/loss: 2.6738, val/MAE: 2.6738, val/MAPE: 7.3238, val/RMSE: 5.0979]
100%|█████████████████████████████████████████| 427/427 [01:24<00:00,  5.03it/s]
2025-06-16 02:26:15,612 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6253, Test MAPE: 6.8107, Test RMSE: 5.0428
2025-06-16 02:26:15,615 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9717, Test MAPE: 8.1741, Test RMSE: 6.0540
2025-06-16 02:26:15,618 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3570, Test MAPE: 9.7617, Test RMSE: 7.0891
2025-06-16 02:26:15,640 - easytorch-training - INFO - Result <test>: [test/time: 84.86 (s), test/loss: 2.7631, test/MAE: 2.9319, test/MAPE: 8.0474, test/RMSE: 6.0004]
2025-06-16 02:26:16,867 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_058.pt saved
2025-06-16 02:26:16,868 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:11:01
2025-06-16 02:26:16,868 - easytorch-training - INFO - Epoch 59 / 100
100%|███████████████████████████████████████| 1498/1498 [06:08<00:00,  4.07it/s]
2025-06-16 02:32:25,321 - easytorch-training - INFO - Result <train>: [train/time: 368.45 (s), train/lr: 1.00e-05, train/loss: 2.5902, train/MAE: 2.5902, train/MAPE: 6.7721, train/RMSE: 5.0853]
2025-06-16 02:32:25,322 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:42<00:00,  5.00it/s]
2025-06-16 02:33:07,930 - easytorch-training - INFO - Result <val>: [val/time: 42.61 (s), val/loss: 2.6733, val/MAE: 2.6733, val/MAPE: 7.3343, val/RMSE: 5.0947]
100%|█████████████████████████████████████████| 427/427 [01:24<00:00,  5.04it/s]
2025-06-16 02:34:32,737 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6264, Test MAPE: 6.8218, Test RMSE: 5.0465
2025-06-16 02:34:32,739 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9724, Test MAPE: 8.1903, Test RMSE: 6.0576
2025-06-16 02:34:32,741 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3603, Test MAPE: 9.8021, Test RMSE: 7.0969
2025-06-16 02:34:32,764 - easytorch-training - INFO - Result <test>: [test/time: 84.83 (s), test/loss: 2.7644, test/MAE: 2.9330, test/MAPE: 8.0724, test/RMSE: 6.0038]
2025-06-16 02:34:34,047 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_059.pt saved
2025-06-16 02:34:34,047 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:11:09
2025-06-16 02:34:34,047 - easytorch-training - INFO - Epoch 60 / 100
100%|███████████████████████████████████████| 1498/1498 [06:07<00:00,  4.07it/s]
2025-06-16 02:40:41,934 - easytorch-training - INFO - Result <train>: [train/time: 367.89 (s), train/lr: 1.00e-05, train/loss: 2.5890, train/MAE: 2.5890, train/MAPE: 6.7646, train/RMSE: 5.0835]
2025-06-16 02:40:41,935 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:42<00:00,  4.99it/s]
2025-06-16 02:41:24,657 - easytorch-training - INFO - Result <val>: [val/time: 42.72 (s), val/loss: 2.6734, val/MAE: 2.6734, val/MAPE: 7.3160, val/RMSE: 5.0933]
100%|█████████████████████████████████████████| 427/427 [01:24<00:00,  5.04it/s]
2025-06-16 02:42:49,471 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6231, Test MAPE: 6.7839, Test RMSE: 5.0332
2025-06-16 02:42:49,474 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9680, Test MAPE: 8.1496, Test RMSE: 6.0371
2025-06-16 02:42:49,476 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3580, Test MAPE: 9.7960, Test RMSE: 7.0891
2025-06-16 02:42:49,500 - easytorch-training - INFO - Result <test>: [test/time: 84.84 (s), test/loss: 2.7614, test/MAE: 2.9299, test/MAPE: 8.0391, test/RMSE: 5.9913]
2025-06-16 02:42:50,842 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_060.pt saved
2025-06-16 02:42:51,785 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:11:18
2025-06-16 02:42:51,785 - easytorch-training - INFO - Epoch 61 / 100
100%|███████████████████████████████████████| 1498/1498 [06:07<00:00,  4.07it/s]
2025-06-16 02:48:59,565 - easytorch-training - INFO - Result <train>: [train/time: 367.78 (s), train/lr: 1.00e-05, train/loss: 2.5891, train/MAE: 2.5891, train/MAPE: 6.7665, train/RMSE: 5.0837]
2025-06-16 02:48:59,566 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:42<00:00,  5.01it/s]
2025-06-16 02:49:42,077 - easytorch-training - INFO - Result <val>: [val/time: 42.51 (s), val/loss: 2.6733, val/MAE: 2.6733, val/MAPE: 7.3356, val/RMSE: 5.0966]
100%|█████████████████████████████████████████| 427/427 [01:24<00:00,  5.03it/s]
2025-06-16 02:51:06,926 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6275, Test MAPE: 6.8212, Test RMSE: 5.0485
2025-06-16 02:51:06,929 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9732, Test MAPE: 8.1888, Test RMSE: 6.0586
2025-06-16 02:51:06,932 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3589, Test MAPE: 9.8137, Test RMSE: 7.0955
2025-06-16 02:51:06,956 - easytorch-training - INFO - Result <test>: [test/time: 84.88 (s), test/loss: 2.7647, test/MAE: 2.9334, test/MAPE: 8.0730, test/RMSE: 6.0049]
2025-06-16 02:51:08,246 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_061.pt saved
2025-06-16 02:51:08,247 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:11:24
2025-06-16 02:51:08,247 - easytorch-training - INFO - Epoch 62 / 100
100%|███████████████████████████████████████| 1498/1498 [06:07<00:00,  4.08it/s]
2025-06-16 02:57:15,757 - easytorch-training - INFO - Result <train>: [train/time: 367.51 (s), train/lr: 1.00e-05, train/loss: 2.5888, train/MAE: 2.5888, train/MAPE: 6.7657, train/RMSE: 5.0821]
2025-06-16 02:57:15,757 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:42<00:00,  5.01it/s]
2025-06-16 02:57:58,312 - easytorch-training - INFO - Result <val>: [val/time: 42.55 (s), val/loss: 2.6741, val/MAE: 2.6741, val/MAPE: 7.2976, val/RMSE: 5.0915]
100%|█████████████████████████████████████████| 427/427 [01:24<00:00,  5.04it/s]
2025-06-16 02:59:22,998 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6237, Test MAPE: 6.7819, Test RMSE: 5.0335
2025-06-16 02:59:23,005 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9683, Test MAPE: 8.1307, Test RMSE: 6.0370
2025-06-16 02:59:23,008 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3540, Test MAPE: 9.7207, Test RMSE: 7.0762
2025-06-16 02:59:23,036 - easytorch-training - INFO - Result <test>: [test/time: 84.72 (s), test/loss: 2.7603, test/MAE: 2.9290, test/MAPE: 8.0139, test/RMSE: 5.9866]
2025-06-16 02:59:24,407 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_062.pt saved
2025-06-16 02:59:24,407 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:11:29
2025-06-16 02:59:24,407 - easytorch-training - INFO - Epoch 63 / 100
100%|███████████████████████████████████████| 1498/1498 [06:07<00:00,  4.07it/s]
2025-06-16 03:05:32,168 - easytorch-training - INFO - Result <train>: [train/time: 367.76 (s), train/lr: 1.00e-05, train/loss: 2.5884, train/MAE: 2.5884, train/MAPE: 6.7636, train/RMSE: 5.0812]
2025-06-16 03:05:32,169 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:42<00:00,  5.00it/s]
2025-06-16 03:06:14,791 - easytorch-training - INFO - Result <val>: [val/time: 42.62 (s), val/loss: 2.6729, val/MAE: 2.6729, val/MAPE: 7.3360, val/RMSE: 5.0955]
100%|█████████████████████████████████████████| 427/427 [01:24<00:00,  5.04it/s]
2025-06-16 03:07:39,519 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6286, Test MAPE: 6.8336, Test RMSE: 5.0507
2025-06-16 03:07:39,521 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9753, Test MAPE: 8.1985, Test RMSE: 6.0643
2025-06-16 03:07:39,523 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3621, Test MAPE: 9.8045, Test RMSE: 7.1048
2025-06-16 03:07:39,546 - easytorch-training - INFO - Result <test>: [test/time: 84.75 (s), test/loss: 2.7670, test/MAE: 2.9353, test/MAPE: 8.0772, test/RMSE: 6.0106]
2025-06-16 03:07:40,809 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_063.pt saved
2025-06-16 03:07:40,809 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:11:35
2025-06-16 03:07:40,809 - easytorch-training - INFO - Epoch 64 / 100
100%|███████████████████████████████████████| 1498/1498 [06:07<00:00,  4.08it/s]
2025-06-16 03:13:47,881 - easytorch-training - INFO - Result <train>: [train/time: 367.07 (s), train/lr: 1.00e-05, train/loss: 2.5885, train/MAE: 2.5885, train/MAPE: 6.7645, train/RMSE: 5.0804]
2025-06-16 03:13:47,883 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:42<00:00,  5.04it/s]
2025-06-16 03:14:30,159 - easytorch-training - INFO - Result <val>: [val/time: 42.28 (s), val/loss: 2.6726, val/MAE: 2.6726, val/MAPE: 7.3155, val/RMSE: 5.0946]
100%|█████████████████████████████████████████| 427/427 [01:24<00:00,  5.08it/s]
2025-06-16 03:15:54,171 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6263, Test MAPE: 6.8108, Test RMSE: 5.0474
2025-06-16 03:15:54,173 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9710, Test MAPE: 8.1691, Test RMSE: 6.0529
2025-06-16 03:15:54,175 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3568, Test MAPE: 9.7492, Test RMSE: 7.0886
2025-06-16 03:15:54,197 - easytorch-training - INFO - Result <test>: [test/time: 84.04 (s), test/loss: 2.7632, test/MAE: 2.9317, test/MAPE: 8.0441, test/RMSE: 6.0007]
2025-06-16 03:15:55,420 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_064.pt saved
2025-06-16 03:15:55,420 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:11:38
2025-06-16 03:15:55,420 - easytorch-training - INFO - Epoch 65 / 100
100%|███████████████████████████████████████| 1498/1498 [06:04<00:00,  4.11it/s]
2025-06-16 03:22:00,127 - easytorch-training - INFO - Result <train>: [train/time: 364.71 (s), train/lr: 1.00e-05, train/loss: 2.5874, train/MAE: 2.5874, train/MAPE: 6.7610, train/RMSE: 5.0791]
2025-06-16 03:22:00,144 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:42<00:00,  5.04it/s]
2025-06-16 03:22:42,373 - easytorch-training - INFO - Result <val>: [val/time: 42.23 (s), val/loss: 2.6725, val/MAE: 2.6725, val/MAPE: 7.2986, val/RMSE: 5.0876]
100%|█████████████████████████████████████████| 427/427 [01:23<00:00,  5.09it/s]
2025-06-16 03:24:06,324 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6231, Test MAPE: 6.7887, Test RMSE: 5.0327
2025-06-16 03:24:06,327 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9673, Test MAPE: 8.1420, Test RMSE: 6.0354
2025-06-16 03:24:06,330 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3535, Test MAPE: 9.7383, Test RMSE: 7.0738
2025-06-16 03:24:06,354 - easytorch-training - INFO - Result <test>: [test/time: 83.98 (s), test/loss: 2.7596, test/MAE: 2.9283, test/MAPE: 8.0234, test/RMSE: 5.9845]
2025-06-16 03:24:07,570 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_065.pt saved
2025-06-16 03:24:07,570 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:11:37
2025-06-16 03:24:07,570 - easytorch-training - INFO - Epoch 66 / 100
100%|███████████████████████████████████████| 1498/1498 [06:04<00:00,  4.11it/s]
2025-06-16 03:30:12,111 - easytorch-training - INFO - Result <train>: [train/time: 364.54 (s), train/lr: 1.00e-05, train/loss: 2.5881, train/MAE: 2.5881, train/MAPE: 6.7623, train/RMSE: 5.0786]
2025-06-16 03:30:12,112 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:42<00:00,  5.04it/s]
2025-06-16 03:30:54,348 - easytorch-training - INFO - Result <val>: [val/time: 42.24 (s), val/loss: 2.6750, val/MAE: 2.6750, val/MAPE: 7.3062, val/RMSE: 5.0995]
100%|█████████████████████████████████████████| 427/427 [01:23<00:00,  5.09it/s]
2025-06-16 03:32:18,222 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6268, Test MAPE: 6.7875, Test RMSE: 5.0462
2025-06-16 03:32:18,225 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9740, Test MAPE: 8.1496, Test RMSE: 6.0589
2025-06-16 03:32:18,227 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3601, Test MAPE: 9.7731, Test RMSE: 7.0967
2025-06-16 03:32:18,249 - easytorch-training - INFO - Result <test>: [test/time: 83.90 (s), test/loss: 2.7647, test/MAE: 2.9337, test/MAPE: 8.0375, test/RMSE: 6.0043]
2025-06-16 03:32:19,438 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_066.pt saved
2025-06-16 03:32:19,439 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:11:35
2025-06-16 03:32:19,439 - easytorch-training - INFO - Epoch 67 / 100
100%|███████████████████████████████████████| 1498/1498 [06:04<00:00,  4.11it/s]
2025-06-16 03:38:24,333 - easytorch-training - INFO - Result <train>: [train/time: 364.89 (s), train/lr: 1.00e-05, train/loss: 2.5871, train/MAE: 2.5871, train/MAPE: 6.7580, train/RMSE: 5.0759]
2025-06-16 03:38:24,348 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:42<00:00,  5.03it/s]
2025-06-16 03:39:06,659 - easytorch-training - INFO - Result <val>: [val/time: 42.31 (s), val/loss: 2.6727, val/MAE: 2.6727, val/MAPE: 7.3111, val/RMSE: 5.0914]
100%|█████████████████████████████████████████| 427/427 [01:23<00:00,  5.09it/s]
2025-06-16 03:40:30,606 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6256, Test MAPE: 6.7875, Test RMSE: 5.0431
2025-06-16 03:40:30,608 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9705, Test MAPE: 8.1534, Test RMSE: 6.0488
2025-06-16 03:40:30,611 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3588, Test MAPE: 9.7828, Test RMSE: 7.0914
2025-06-16 03:40:30,633 - easytorch-training - INFO - Result <test>: [test/time: 83.97 (s), test/loss: 2.7631, test/MAE: 2.9315, test/MAPE: 8.0415, test/RMSE: 5.9974]
2025-06-16 03:40:31,837 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_067.pt saved
2025-06-16 03:40:31,837 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:11:35
2025-06-16 03:40:31,837 - easytorch-training - INFO - Epoch 68 / 100
100%|███████████████████████████████████████| 1498/1498 [06:05<00:00,  4.10it/s]
2025-06-16 03:46:36,947 - easytorch-training - INFO - Result <train>: [train/time: 365.11 (s), train/lr: 1.00e-05, train/loss: 2.5857, train/MAE: 2.5857, train/MAPE: 6.7541, train/RMSE: 5.0745]
2025-06-16 03:46:36,948 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:42<00:00,  5.05it/s]
2025-06-16 03:47:19,152 - easytorch-training - INFO - Result <val>: [val/time: 42.20 (s), val/loss: 2.6735, val/MAE: 2.6735, val/MAPE: 7.3271, val/RMSE: 5.0972]
100%|█████████████████████████████████████████| 427/427 [01:24<00:00,  5.08it/s]
2025-06-16 03:48:43,269 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6284, Test MAPE: 6.8146, Test RMSE: 5.0515
2025-06-16 03:48:43,271 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9741, Test MAPE: 8.1853, Test RMSE: 6.0601
2025-06-16 03:48:43,274 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3604, Test MAPE: 9.7886, Test RMSE: 7.0980
2025-06-16 03:48:43,297 - easytorch-training - INFO - Result <test>: [test/time: 84.14 (s), test/loss: 2.7652, test/MAE: 2.9341, test/MAPE: 8.0612, test/RMSE: 6.0056]
2025-06-16 03:48:44,493 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_068.pt saved
2025-06-16 03:48:44,493 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:11:35
2025-06-16 03:48:44,493 - easytorch-training - INFO - Epoch 69 / 100
100%|███████████████████████████████████████| 1498/1498 [06:05<00:00,  4.10it/s]
2025-06-16 03:54:49,579 - easytorch-training - INFO - Result <train>: [train/time: 365.09 (s), train/lr: 1.00e-05, train/loss: 2.5868, train/MAE: 2.5868, train/MAPE: 6.7596, train/RMSE: 5.0777]
2025-06-16 03:54:49,580 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:42<00:00,  5.03it/s]
2025-06-16 03:55:31,943 - easytorch-training - INFO - Result <val>: [val/time: 42.36 (s), val/loss: 2.6758, val/MAE: 2.6758, val/MAPE: 7.2863, val/RMSE: 5.1021]
100%|█████████████████████████████████████████| 427/427 [01:24<00:00,  5.07it/s]
2025-06-16 03:56:56,105 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6270, Test MAPE: 6.7705, Test RMSE: 5.0442
2025-06-16 03:56:56,107 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9745, Test MAPE: 8.1156, Test RMSE: 6.0555
2025-06-16 03:56:56,109 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3627, Test MAPE: 9.7364, Test RMSE: 7.1019
2025-06-16 03:56:56,131 - easytorch-training - INFO - Result <test>: [test/time: 84.19 (s), test/loss: 2.7662, test/MAE: 2.9350, test/MAPE: 8.0096, test/RMSE: 6.0054]
2025-06-16 03:56:57,320 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_069.pt saved
2025-06-16 03:56:57,320 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:11:35
2025-06-16 03:56:57,320 - easytorch-training - INFO - Epoch 70 / 100
100%|███████████████████████████████████████| 1498/1498 [06:04<00:00,  4.11it/s]
2025-06-16 04:03:01,890 - easytorch-training - INFO - Result <train>: [train/time: 364.57 (s), train/lr: 1.00e-05, train/loss: 2.5858, train/MAE: 2.5858, train/MAPE: 6.7535, train/RMSE: 5.0729]
2025-06-16 04:03:01,891 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:42<00:00,  5.04it/s]
2025-06-16 04:03:44,154 - easytorch-training - INFO - Result <val>: [val/time: 42.26 (s), val/loss: 2.6765, val/MAE: 2.6765, val/MAPE: 7.3381, val/RMSE: 5.1037]
100%|█████████████████████████████████████████| 427/427 [01:24<00:00,  5.08it/s]
2025-06-16 04:05:08,276 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6281, Test MAPE: 6.8063, Test RMSE: 5.0481
2025-06-16 04:05:08,278 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9750, Test MAPE: 8.1797, Test RMSE: 6.0622
2025-06-16 04:05:08,280 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3612, Test MAPE: 9.8115, Test RMSE: 7.1049
2025-06-16 04:05:08,306 - easytorch-training - INFO - Result <test>: [test/time: 84.15 (s), test/loss: 2.7663, test/MAE: 2.9349, test/MAPE: 8.0635, test/RMSE: 6.0094]
2025-06-16 04:05:09,605 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_070.pt saved
2025-06-16 04:05:10,324 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:11:35
2025-06-16 04:05:10,324 - easytorch-training - INFO - Epoch 71 / 100
100%|███████████████████████████████████████| 1498/1498 [06:04<00:00,  4.11it/s]
2025-06-16 04:11:14,409 - easytorch-training - INFO - Result <train>: [train/time: 364.08 (s), train/lr: 1.00e-05, train/loss: 2.5849, train/MAE: 2.5849, train/MAPE: 6.7491, train/RMSE: 5.0698]
2025-06-16 04:11:14,410 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:42<00:00,  5.05it/s]
2025-06-16 04:11:56,619 - easytorch-training - INFO - Result <val>: [val/time: 42.21 (s), val/loss: 2.6744, val/MAE: 2.6744, val/MAPE: 7.3228, val/RMSE: 5.0975]
100%|█████████████████████████████████████████| 427/427 [01:23<00:00,  5.08it/s]
2025-06-16 04:13:20,611 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6263, Test MAPE: 6.8080, Test RMSE: 5.0426
2025-06-16 04:13:20,612 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9727, Test MAPE: 8.1760, Test RMSE: 6.0527
2025-06-16 04:13:20,615 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3609, Test MAPE: 9.7624, Test RMSE: 7.0983
2025-06-16 04:13:20,640 - easytorch-training - INFO - Result <test>: [test/time: 84.02 (s), test/loss: 2.7646, test/MAE: 2.9332, test/MAPE: 8.0509, test/RMSE: 6.0012]
2025-06-16 04:13:21,853 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_071.pt saved
2025-06-16 04:13:21,854 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:11:33
2025-06-16 04:13:21,854 - easytorch-training - INFO - Epoch 72 / 100
100%|███████████████████████████████████████| 1498/1498 [06:03<00:00,  4.12it/s]
2025-06-16 04:19:25,698 - easytorch-training - INFO - Result <train>: [train/time: 363.84 (s), train/lr: 1.00e-05, train/loss: 2.5854, train/MAE: 2.5854, train/MAPE: 6.7540, train/RMSE: 5.0730]
2025-06-16 04:19:25,700 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:42<00:00,  5.06it/s]
2025-06-16 04:20:07,760 - easytorch-training - INFO - Result <val>: [val/time: 42.06 (s), val/loss: 2.6738, val/MAE: 2.6738, val/MAPE: 7.3407, val/RMSE: 5.0961]
100%|█████████████████████████████████████████| 427/427 [01:23<00:00,  5.11it/s]
2025-06-16 04:21:31,338 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6266, Test MAPE: 6.8308, Test RMSE: 5.0452
2025-06-16 04:21:31,340 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9726, Test MAPE: 8.1933, Test RMSE: 6.0556
2025-06-16 04:21:31,343 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3590, Test MAPE: 9.7963, Test RMSE: 7.0919
2025-06-16 04:21:31,368 - easytorch-training - INFO - Result <test>: [test/time: 83.61 (s), test/loss: 2.7643, test/MAE: 2.9330, test/MAPE: 8.0726, test/RMSE: 6.0021]
2025-06-16 04:21:32,585 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_072.pt saved
2025-06-16 04:21:32,585 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:11:30
2025-06-16 04:21:32,585 - easytorch-training - INFO - Epoch 73 / 100
100%|███████████████████████████████████████| 1498/1498 [06:02<00:00,  4.13it/s]
2025-06-16 04:27:34,970 - easytorch-training - INFO - Result <train>: [train/time: 362.38 (s), train/lr: 1.00e-05, train/loss: 2.5840, train/MAE: 2.5840, train/MAPE: 6.7478, train/RMSE: 5.0678]
2025-06-16 04:27:34,971 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:41<00:00,  5.08it/s]
2025-06-16 04:28:16,933 - easytorch-training - INFO - Result <val>: [val/time: 41.96 (s), val/loss: 2.6760, val/MAE: 2.6760, val/MAPE: 7.3511, val/RMSE: 5.0997]
100%|█████████████████████████████████████████| 427/427 [01:23<00:00,  5.11it/s]
2025-06-16 04:29:40,528 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6276, Test MAPE: 6.8305, Test RMSE: 5.0441
2025-06-16 04:29:40,530 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9749, Test MAPE: 8.2111, Test RMSE: 6.0579
2025-06-16 04:29:40,533 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3628, Test MAPE: 9.8115, Test RMSE: 7.1023
2025-06-16 04:29:40,558 - easytorch-training - INFO - Result <test>: [test/time: 83.62 (s), test/loss: 2.7663, test/MAE: 2.9353, test/MAPE: 8.0830, test/RMSE: 6.0060]
2025-06-16 04:29:41,723 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_073.pt saved
2025-06-16 04:29:41,723 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:11:25
2025-06-16 04:29:41,723 - easytorch-training - INFO - Epoch 74 / 100
100%|███████████████████████████████████████| 1498/1498 [06:02<00:00,  4.13it/s]
2025-06-16 04:35:44,205 - easytorch-training - INFO - Result <train>: [train/time: 362.48 (s), train/lr: 1.00e-05, train/loss: 2.5846, train/MAE: 2.5846, train/MAPE: 6.7495, train/RMSE: 5.0718]
2025-06-16 04:35:44,205 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:42<00:00,  5.06it/s]
2025-06-16 04:36:26,294 - easytorch-training - INFO - Result <val>: [val/time: 42.09 (s), val/loss: 2.6748, val/MAE: 2.6748, val/MAPE: 7.3435, val/RMSE: 5.0988]
100%|█████████████████████████████████████████| 427/427 [01:23<00:00,  5.11it/s]
2025-06-16 04:37:49,919 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6267, Test MAPE: 6.8183, Test RMSE: 5.0473
2025-06-16 04:37:49,921 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9723, Test MAPE: 8.1921, Test RMSE: 6.0578
2025-06-16 04:37:49,923 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3593, Test MAPE: 9.8175, Test RMSE: 7.0984
2025-06-16 04:37:49,945 - easytorch-training - INFO - Result <test>: [test/time: 83.65 (s), test/loss: 2.7645, test/MAE: 2.9327, test/MAPE: 8.0748, test/RMSE: 6.0046]
2025-06-16 04:37:51,110 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_074.pt saved
2025-06-16 04:37:51,110 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:11:21
2025-06-16 04:37:51,110 - easytorch-training - INFO - Epoch 75 / 100
100%|███████████████████████████████████████| 1498/1498 [06:02<00:00,  4.13it/s]
2025-06-16 04:43:53,685 - easytorch-training - INFO - Result <train>: [train/time: 362.57 (s), train/lr: 1.00e-05, train/loss: 2.5836, train/MAE: 2.5836, train/MAPE: 6.7453, train/RMSE: 5.0677]
2025-06-16 04:43:53,686 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:42<00:00,  5.07it/s]
2025-06-16 04:44:35,688 - easytorch-training - INFO - Result <val>: [val/time: 42.00 (s), val/loss: 2.6744, val/MAE: 2.6744, val/MAPE: 7.3504, val/RMSE: 5.1016]
100%|█████████████████████████████████████████| 427/427 [01:23<00:00,  5.11it/s]
2025-06-16 04:45:59,204 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6274, Test MAPE: 6.8243, Test RMSE: 5.0488
2025-06-16 04:45:59,206 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9746, Test MAPE: 8.2097, Test RMSE: 6.0629
2025-06-16 04:45:59,207 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3627, Test MAPE: 9.8286, Test RMSE: 7.1077
2025-06-16 04:45:59,231 - easytorch-training - INFO - Result <test>: [test/time: 83.54 (s), test/loss: 2.7662, test/MAE: 2.9350, test/MAPE: 8.0836, test/RMSE: 6.0109]
2025-06-16 04:46:00,402 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_075.pt saved
2025-06-16 04:46:00,403 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:11:16
2025-06-16 04:46:00,403 - easytorch-training - INFO - Epoch 76 / 100
100%|███████████████████████████████████████| 1498/1498 [06:01<00:00,  4.14it/s]
2025-06-16 04:52:02,028 - easytorch-training - INFO - Result <train>: [train/time: 361.62 (s), train/lr: 1.00e-05, train/loss: 2.5834, train/MAE: 2.5834, train/MAPE: 6.7461, train/RMSE: 5.0692]
2025-06-16 04:52:02,029 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:41<00:00,  5.11it/s]
2025-06-16 04:52:43,716 - easytorch-training - INFO - Result <val>: [val/time: 41.69 (s), val/loss: 2.6754, val/MAE: 2.6754, val/MAPE: 7.3265, val/RMSE: 5.1021]
100%|█████████████████████████████████████████| 427/427 [01:23<00:00,  5.14it/s]
2025-06-16 04:54:06,723 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6278, Test MAPE: 6.8019, Test RMSE: 5.0495
2025-06-16 04:54:06,725 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9748, Test MAPE: 8.1799, Test RMSE: 6.0636
2025-06-16 04:54:06,727 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3645, Test MAPE: 9.8017, Test RMSE: 7.1120
2025-06-16 04:54:06,750 - easytorch-training - INFO - Result <test>: [test/time: 83.03 (s), test/loss: 2.7671, test/MAE: 2.9357, test/MAPE: 8.0606, test/RMSE: 6.0127]
2025-06-16 04:54:07,915 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_076.pt saved
2025-06-16 04:54:07,915 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:11:09
2025-06-16 04:54:07,915 - easytorch-training - INFO - Epoch 77 / 100
100%|███████████████████████████████████████| 1498/1498 [06:00<00:00,  4.15it/s]
2025-06-16 05:00:08,564 - easytorch-training - INFO - Result <train>: [train/time: 360.65 (s), train/lr: 1.00e-05, train/loss: 2.5828, train/MAE: 2.5828, train/MAPE: 6.7428, train/RMSE: 5.0648]
2025-06-16 05:00:08,565 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:41<00:00,  5.11it/s]
2025-06-16 05:00:50,272 - easytorch-training - INFO - Result <val>: [val/time: 41.71 (s), val/loss: 2.6762, val/MAE: 2.6762, val/MAPE: 7.3472, val/RMSE: 5.1072]
100%|█████████████████████████████████████████| 427/427 [01:22<00:00,  5.15it/s]
2025-06-16 05:02:13,221 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6296, Test MAPE: 6.8265, Test RMSE: 5.0582
2025-06-16 05:02:13,223 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9781, Test MAPE: 8.2012, Test RMSE: 6.0772
2025-06-16 05:02:13,225 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3661, Test MAPE: 9.8185, Test RMSE: 7.1187
2025-06-16 05:02:13,247 - easytorch-training - INFO - Result <test>: [test/time: 82.97 (s), test/loss: 2.7692, test/MAE: 2.9379, test/MAPE: 8.0817, test/RMSE: 6.0226]
2025-06-16 05:02:14,404 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_077.pt saved
2025-06-16 05:02:14,404 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:11:01
2025-06-16 05:02:14,404 - easytorch-training - INFO - Epoch 78 / 100
100%|███████████████████████████████████████| 1498/1498 [06:00<00:00,  4.15it/s]
2025-06-16 05:08:15,007 - easytorch-training - INFO - Result <train>: [train/time: 360.60 (s), train/lr: 1.00e-05, train/loss: 2.5828, train/MAE: 2.5828, train/MAPE: 6.7419, train/RMSE: 5.0658]
2025-06-16 05:08:15,009 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:41<00:00,  5.11it/s]
2025-06-16 05:08:56,695 - easytorch-training - INFO - Result <val>: [val/time: 41.69 (s), val/loss: 2.6747, val/MAE: 2.6747, val/MAPE: 7.3438, val/RMSE: 5.0982]
100%|█████████████████████████████████████████| 427/427 [01:22<00:00,  5.15it/s]
2025-06-16 05:10:19,663 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6262, Test MAPE: 6.8185, Test RMSE: 5.0436
2025-06-16 05:10:19,665 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9734, Test MAPE: 8.1904, Test RMSE: 6.0585
2025-06-16 05:10:19,667 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3622, Test MAPE: 9.8205, Test RMSE: 7.1044
2025-06-16 05:10:19,691 - easytorch-training - INFO - Result <test>: [test/time: 82.99 (s), test/loss: 2.7652, test/MAE: 2.9340, test/MAPE: 8.0751, test/RMSE: 6.0064]
2025-06-16 05:10:20,821 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_078.pt saved
2025-06-16 05:10:20,822 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:10:54
2025-06-16 05:10:20,822 - easytorch-training - INFO - Epoch 79 / 100
100%|███████████████████████████████████████| 1498/1498 [06:00<00:00,  4.15it/s]
2025-06-16 05:16:21,395 - easytorch-training - INFO - Result <train>: [train/time: 360.57 (s), train/lr: 1.00e-05, train/loss: 2.5826, train/MAE: 2.5826, train/MAPE: 6.7398, train/RMSE: 5.0656]
2025-06-16 05:16:21,397 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:41<00:00,  5.11it/s]
2025-06-16 05:17:03,107 - easytorch-training - INFO - Result <val>: [val/time: 41.71 (s), val/loss: 2.6754, val/MAE: 2.6754, val/MAPE: 7.3725, val/RMSE: 5.1010]
100%|█████████████████████████████████████████| 427/427 [01:22<00:00,  5.15it/s]
2025-06-16 05:18:26,075 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6280, Test MAPE: 6.8511, Test RMSE: 5.0493
2025-06-16 05:18:26,077 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9748, Test MAPE: 8.2231, Test RMSE: 6.0644
2025-06-16 05:18:26,079 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3643, Test MAPE: 9.8707, Test RMSE: 7.1138
2025-06-16 05:18:26,102 - easytorch-training - INFO - Result <test>: [test/time: 82.99 (s), test/loss: 2.7673, test/MAE: 2.9358, test/MAPE: 8.1105, test/RMSE: 6.0140]
2025-06-16 05:18:27,263 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_079.pt saved
2025-06-16 05:18:27,263 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:10:46
2025-06-16 05:18:27,263 - easytorch-training - INFO - Epoch 80 / 100
100%|███████████████████████████████████████| 1498/1498 [06:00<00:00,  4.16it/s]
2025-06-16 05:24:27,793 - easytorch-training - INFO - Result <train>: [train/time: 360.53 (s), train/lr: 1.00e-05, train/loss: 2.5820, train/MAE: 2.5820, train/MAPE: 6.7429, train/RMSE: 5.0640]
2025-06-16 05:24:27,794 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:41<00:00,  5.11it/s]
2025-06-16 05:25:09,510 - easytorch-training - INFO - Result <val>: [val/time: 41.72 (s), val/loss: 2.6753, val/MAE: 2.6753, val/MAPE: 7.2816, val/RMSE: 5.0968]
100%|█████████████████████████████████████████| 427/427 [01:22<00:00,  5.15it/s]
2025-06-16 05:26:32,471 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6247, Test MAPE: 6.7630, Test RMSE: 5.0380
2025-06-16 05:26:32,473 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9718, Test MAPE: 8.1131, Test RMSE: 6.0475
2025-06-16 05:26:32,475 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3570, Test MAPE: 9.7074, Test RMSE: 7.0825
2025-06-16 05:26:32,497 - easytorch-training - INFO - Result <test>: [test/time: 82.99 (s), test/loss: 2.7626, test/MAE: 2.9315, test/MAPE: 7.9966, test/RMSE: 5.9941]
2025-06-16 05:26:33,649 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_080.pt saved
2025-06-16 05:26:34,851 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:10:40
2025-06-16 05:26:34,851 - easytorch-training - INFO - Epoch 81 / 100
100%|███████████████████████████████████████| 1498/1498 [06:00<00:00,  4.16it/s]
2025-06-16 05:32:35,311 - easytorch-training - INFO - Result <train>: [train/time: 360.46 (s), train/lr: 1.00e-05, train/loss: 2.5825, train/MAE: 2.5825, train/MAPE: 6.7413, train/RMSE: 5.0648]
2025-06-16 05:32:35,312 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:41<00:00,  5.11it/s]
2025-06-16 05:33:16,960 - easytorch-training - INFO - Result <val>: [val/time: 41.65 (s), val/loss: 2.6771, val/MAE: 2.6771, val/MAPE: 7.3491, val/RMSE: 5.1030]
100%|█████████████████████████████████████████| 427/427 [01:22<00:00,  5.15it/s]
2025-06-16 05:34:39,876 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6280, Test MAPE: 6.8172, Test RMSE: 5.0480
2025-06-16 05:34:39,878 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9750, Test MAPE: 8.1898, Test RMSE: 6.0600
2025-06-16 05:34:39,880 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3625, Test MAPE: 9.8173, Test RMSE: 7.1037
2025-06-16 05:34:39,904 - easytorch-training - INFO - Result <test>: [test/time: 82.94 (s), test/loss: 2.7665, test/MAE: 2.9354, test/MAPE: 8.0714, test/RMSE: 6.0078]
2025-06-16 05:34:41,063 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_081.pt saved
2025-06-16 05:34:41,063 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:10:33
2025-06-16 05:34:41,064 - easytorch-training - INFO - Epoch 82 / 100
100%|███████████████████████████████████████| 1498/1498 [06:00<00:00,  4.16it/s]
2025-06-16 05:40:41,302 - easytorch-training - INFO - Result <train>: [train/time: 360.24 (s), train/lr: 1.00e-05, train/loss: 2.5815, train/MAE: 2.5815, train/MAPE: 6.7389, train/RMSE: 5.0625]
2025-06-16 05:40:41,303 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:41<00:00,  5.12it/s]
2025-06-16 05:41:22,924 - easytorch-training - INFO - Result <val>: [val/time: 41.62 (s), val/loss: 2.6774, val/MAE: 2.6774, val/MAPE: 7.3193, val/RMSE: 5.1043]
100%|█████████████████████████████████████████| 427/427 [01:22<00:00,  5.15it/s]
2025-06-16 05:42:45,811 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6277, Test MAPE: 6.7940, Test RMSE: 5.0485
2025-06-16 05:42:45,813 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9743, Test MAPE: 8.1524, Test RMSE: 6.0588
2025-06-16 05:42:45,815 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3634, Test MAPE: 9.7718, Test RMSE: 7.1095
2025-06-16 05:42:45,838 - easytorch-training - INFO - Result <test>: [test/time: 82.91 (s), test/loss: 2.7667, test/MAE: 2.9352, test/MAPE: 8.0402, test/RMSE: 6.0095]
2025-06-16 05:42:46,991 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_082.pt saved
2025-06-16 05:42:46,991 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:10:25
2025-06-16 05:42:46,991 - easytorch-training - INFO - Epoch 83 / 100
100%|███████████████████████████████████████| 1498/1498 [06:00<00:00,  4.16it/s]
2025-06-16 05:48:47,111 - easytorch-training - INFO - Result <train>: [train/time: 360.12 (s), train/lr: 1.00e-05, train/loss: 2.5806, train/MAE: 2.5806, train/MAPE: 6.7378, train/RMSE: 5.0610]
2025-06-16 05:48:47,112 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:41<00:00,  5.11it/s]
2025-06-16 05:49:28,787 - easytorch-training - INFO - Result <val>: [val/time: 41.68 (s), val/loss: 2.6755, val/MAE: 2.6755, val/MAPE: 7.3237, val/RMSE: 5.1057]
100%|█████████████████████████████████████████| 427/427 [01:22<00:00,  5.15it/s]
2025-06-16 05:50:51,665 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6276, Test MAPE: 6.8138, Test RMSE: 5.0503
2025-06-16 05:50:51,668 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9748, Test MAPE: 8.1755, Test RMSE: 6.0638
2025-06-16 05:50:51,670 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3632, Test MAPE: 9.7901, Test RMSE: 7.1103
2025-06-16 05:50:51,692 - easytorch-training - INFO - Result <test>: [test/time: 82.90 (s), test/loss: 2.7664, test/MAE: 2.9354, test/MAPE: 8.0570, test/RMSE: 6.0127]
2025-06-16 05:50:52,838 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_083.pt saved
2025-06-16 05:50:52,838 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:10:17
2025-06-16 05:50:52,838 - easytorch-training - INFO - Epoch 84 / 100
100%|███████████████████████████████████████| 1498/1498 [06:00<00:00,  4.16it/s]
2025-06-16 05:56:53,203 - easytorch-training - INFO - Result <train>: [train/time: 360.36 (s), train/lr: 1.00e-05, train/loss: 2.5804, train/MAE: 2.5804, train/MAPE: 6.7350, train/RMSE: 5.0597]
2025-06-16 05:56:53,204 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:41<00:00,  5.11it/s]
2025-06-16 05:57:34,883 - easytorch-training - INFO - Result <val>: [val/time: 41.68 (s), val/loss: 2.6759, val/MAE: 2.6759, val/MAPE: 7.3327, val/RMSE: 5.1038]
100%|█████████████████████████████████████████| 427/427 [01:22<00:00,  5.15it/s]
2025-06-16 05:58:57,832 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6274, Test MAPE: 6.8190, Test RMSE: 5.0504
2025-06-16 05:58:57,835 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9747, Test MAPE: 8.1848, Test RMSE: 6.0667
2025-06-16 05:58:57,837 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3626, Test MAPE: 9.7964, Test RMSE: 7.1102
2025-06-16 05:58:57,859 - easytorch-training - INFO - Result <test>: [test/time: 82.98 (s), test/loss: 2.7663, test/MAE: 2.9353, test/MAPE: 8.0641, test/RMSE: 6.0139]
2025-06-16 05:58:59,004 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_084.pt saved
2025-06-16 05:58:59,004 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:10:10
2025-06-16 05:58:59,004 - easytorch-training - INFO - Epoch 85 / 100
100%|███████████████████████████████████████| 1498/1498 [06:00<00:00,  4.15it/s]
2025-06-16 06:04:59,928 - easytorch-training - INFO - Result <train>: [train/time: 360.92 (s), train/lr: 1.00e-05, train/loss: 2.5807, train/MAE: 2.5807, train/MAPE: 6.7348, train/RMSE: 5.0604]
2025-06-16 06:04:59,930 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:41<00:00,  5.10it/s]
2025-06-16 06:05:41,668 - easytorch-training - INFO - Result <val>: [val/time: 41.74 (s), val/loss: 2.6774, val/MAE: 2.6774, val/MAPE: 7.3424, val/RMSE: 5.1034]
100%|█████████████████████████████████████████| 427/427 [01:23<00:00,  5.14it/s]
2025-06-16 06:07:04,687 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6289, Test MAPE: 6.8318, Test RMSE: 5.0489
2025-06-16 06:07:04,690 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9763, Test MAPE: 8.2027, Test RMSE: 6.0619
2025-06-16 06:07:04,692 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3639, Test MAPE: 9.7888, Test RMSE: 7.1072
2025-06-16 06:07:04,714 - easytorch-training - INFO - Result <test>: [test/time: 83.04 (s), test/loss: 2.7673, test/MAE: 2.9364, test/MAPE: 8.0692, test/RMSE: 6.0102]
2025-06-16 06:07:05,869 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_085.pt saved
2025-06-16 06:07:05,869 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:10:04
2025-06-16 06:07:05,869 - easytorch-training - INFO - Epoch 86 / 100
100%|███████████████████████████████████████| 1498/1498 [06:01<00:00,  4.15it/s]
2025-06-16 06:13:06,882 - easytorch-training - INFO - Result <train>: [train/time: 361.01 (s), train/lr: 1.00e-05, train/loss: 2.5802, train/MAE: 2.5802, train/MAPE: 6.7348, train/RMSE: 5.0580]
2025-06-16 06:13:06,884 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:41<00:00,  5.11it/s]
2025-06-16 06:13:48,597 - easytorch-training - INFO - Result <val>: [val/time: 41.71 (s), val/loss: 2.6739, val/MAE: 2.6739, val/MAPE: 7.3383, val/RMSE: 5.0997]
100%|█████████████████████████████████████████| 427/427 [01:22<00:00,  5.15it/s]
2025-06-16 06:15:11,526 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6270, Test MAPE: 6.8108, Test RMSE: 5.0479
2025-06-16 06:15:11,528 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9734, Test MAPE: 8.1814, Test RMSE: 6.0608
2025-06-16 06:15:11,531 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3629, Test MAPE: 9.8127, Test RMSE: 7.1092
2025-06-16 06:15:11,556 - easytorch-training - INFO - Result <test>: [test/time: 82.96 (s), test/loss: 2.7657, test/MAE: 2.9345, test/MAPE: 8.0653, test/RMSE: 6.0103]
2025-06-16 06:15:12,698 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_086.pt saved
2025-06-16 06:15:12,698 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:09:58
2025-06-16 06:15:12,698 - easytorch-training - INFO - Epoch 87 / 100
100%|███████████████████████████████████████| 1498/1498 [06:00<00:00,  4.16it/s]
2025-06-16 06:21:13,166 - easytorch-training - INFO - Result <train>: [train/time: 360.47 (s), train/lr: 1.00e-05, train/loss: 2.5801, train/MAE: 2.5801, train/MAPE: 6.7321, train/RMSE: 5.0583]
2025-06-16 06:21:13,168 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:41<00:00,  5.11it/s]
2025-06-16 06:21:54,824 - easytorch-training - INFO - Result <val>: [val/time: 41.66 (s), val/loss: 2.6768, val/MAE: 2.6768, val/MAPE: 7.3376, val/RMSE: 5.1033]
100%|█████████████████████████████████████████| 427/427 [01:22<00:00,  5.15it/s]
2025-06-16 06:23:17,784 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6290, Test MAPE: 6.8349, Test RMSE: 5.0520
2025-06-16 06:23:17,786 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9769, Test MAPE: 8.1869, Test RMSE: 6.0667
2025-06-16 06:23:17,788 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3656, Test MAPE: 9.7949, Test RMSE: 7.1122
2025-06-16 06:23:17,811 - easytorch-training - INFO - Result <test>: [test/time: 82.99 (s), test/loss: 2.7686, test/MAE: 2.9371, test/MAPE: 8.0698, test/RMSE: 6.0143]
2025-06-16 06:23:19,002 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_087.pt saved
2025-06-16 06:23:19,002 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:09:52
2025-06-16 06:23:19,002 - easytorch-training - INFO - Epoch 88 / 100
100%|███████████████████████████████████████| 1498/1498 [06:00<00:00,  4.16it/s]
2025-06-16 06:29:19,203 - easytorch-training - INFO - Result <train>: [train/time: 360.20 (s), train/lr: 1.00e-05, train/loss: 2.5798, train/MAE: 2.5798, train/MAPE: 6.7316, train/RMSE: 5.0552]
2025-06-16 06:29:19,205 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:41<00:00,  5.12it/s]
2025-06-16 06:30:00,842 - easytorch-training - INFO - Result <val>: [val/time: 41.64 (s), val/loss: 2.6755, val/MAE: 2.6755, val/MAPE: 7.3369, val/RMSE: 5.1019]
100%|█████████████████████████████████████████| 427/427 [01:22<00:00,  5.15it/s]
2025-06-16 06:31:23,758 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6294, Test MAPE: 6.8288, Test RMSE: 5.0540
2025-06-16 06:31:23,760 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9770, Test MAPE: 8.1985, Test RMSE: 6.0689
2025-06-16 06:31:23,762 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3646, Test MAPE: 9.7806, Test RMSE: 7.1113
2025-06-16 06:31:23,783 - easytorch-training - INFO - Result <test>: [test/time: 82.94 (s), test/loss: 2.7681, test/MAE: 2.9372, test/MAPE: 8.0714, test/RMSE: 6.0162]
2025-06-16 06:31:24,904 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_088.pt saved
2025-06-16 06:31:24,904 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:09:45
2025-06-16 06:31:24,904 - easytorch-training - INFO - Epoch 89 / 100
100%|███████████████████████████████████████| 1498/1498 [06:00<00:00,  4.16it/s]
2025-06-16 06:37:25,216 - easytorch-training - INFO - Result <train>: [train/time: 360.31 (s), train/lr: 1.00e-05, train/loss: 2.5794, train/MAE: 2.5794, train/MAPE: 6.7301, train/RMSE: 5.0557]
2025-06-16 06:37:25,218 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:41<00:00,  5.11it/s]
2025-06-16 06:38:06,898 - easytorch-training - INFO - Result <val>: [val/time: 41.68 (s), val/loss: 2.6773, val/MAE: 2.6773, val/MAPE: 7.3162, val/RMSE: 5.1050]
100%|█████████████████████████████████████████| 427/427 [01:22<00:00,  5.15it/s]
2025-06-16 06:39:29,792 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6297, Test MAPE: 6.8084, Test RMSE: 5.0547
2025-06-16 06:39:29,794 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9778, Test MAPE: 8.1636, Test RMSE: 6.0675
2025-06-16 06:39:29,796 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3662, Test MAPE: 9.7471, Test RMSE: 7.1132
2025-06-16 06:39:29,818 - easytorch-training - INFO - Result <test>: [test/time: 82.92 (s), test/loss: 2.7694, test/MAE: 2.9383, test/MAPE: 8.0416, test/RMSE: 6.0169]
2025-06-16 06:39:30,987 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_089.pt saved
2025-06-16 06:39:30,988 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:09:39
2025-06-16 06:39:30,988 - easytorch-training - INFO - Epoch 90 / 100
100%|███████████████████████████████████████| 1498/1498 [06:00<00:00,  4.16it/s]
2025-06-16 06:45:31,338 - easytorch-training - INFO - Result <train>: [train/time: 360.35 (s), train/lr: 1.00e-05, train/loss: 2.5788, train/MAE: 2.5788, train/MAPE: 6.7289, train/RMSE: 5.0540]
2025-06-16 06:45:31,339 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:41<00:00,  5.11it/s]
2025-06-16 06:46:13,049 - easytorch-training - INFO - Result <val>: [val/time: 41.71 (s), val/loss: 2.6777, val/MAE: 2.6777, val/MAPE: 7.3102, val/RMSE: 5.1049]
100%|█████████████████████████████████████████| 427/427 [01:22<00:00,  5.14it/s]
2025-06-16 06:47:36,053 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6272, Test MAPE: 6.7896, Test RMSE: 5.0460
2025-06-16 06:47:36,055 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9757, Test MAPE: 8.1507, Test RMSE: 6.0616
2025-06-16 06:47:36,058 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3670, Test MAPE: 9.7468, Test RMSE: 7.1153
2025-06-16 06:47:36,080 - easytorch-training - INFO - Result <test>: [test/time: 83.03 (s), test/loss: 2.7680, test/MAE: 2.9368, test/MAPE: 8.0317, test/RMSE: 6.0125]
2025-06-16 06:47:37,245 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_090.pt saved
2025-06-16 06:47:38,468 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:09:34
2025-06-16 06:47:38,468 - easytorch-training - INFO - Epoch 91 / 100
100%|███████████████████████████████████████| 1498/1498 [06:00<00:00,  4.16it/s]
2025-06-16 06:53:38,941 - easytorch-training - INFO - Result <train>: [train/time: 360.47 (s), train/lr: 1.00e-05, train/loss: 2.5783, train/MAE: 2.5783, train/MAPE: 6.7264, train/RMSE: 5.0530]
2025-06-16 06:53:38,942 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:41<00:00,  5.11it/s]
2025-06-16 06:54:20,634 - easytorch-training - INFO - Result <val>: [val/time: 41.69 (s), val/loss: 2.6789, val/MAE: 2.6789, val/MAPE: 7.3477, val/RMSE: 5.1093]
100%|█████████████████████████████████████████| 427/427 [01:22<00:00,  5.15it/s]
2025-06-16 06:55:43,547 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6305, Test MAPE: 6.8311, Test RMSE: 5.0543
2025-06-16 06:55:43,549 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9787, Test MAPE: 8.1982, Test RMSE: 6.0716
2025-06-16 06:55:43,551 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3642, Test MAPE: 9.7967, Test RMSE: 7.1097
2025-06-16 06:55:43,574 - easytorch-training - INFO - Result <test>: [test/time: 82.94 (s), test/loss: 2.7690, test/MAE: 2.9382, test/MAPE: 8.0756, test/RMSE: 6.0174]
2025-06-16 06:55:44,763 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_091.pt saved
2025-06-16 06:55:44,764 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:09:28
2025-06-16 06:55:44,764 - easytorch-training - INFO - Epoch 92 / 100
100%|███████████████████████████████████████| 1498/1498 [06:00<00:00,  4.16it/s]
2025-06-16 07:01:45,081 - easytorch-training - INFO - Result <train>: [train/time: 360.32 (s), train/lr: 1.00e-05, train/loss: 2.5781, train/MAE: 2.5781, train/MAPE: 6.7264, train/RMSE: 5.0525]
2025-06-16 07:01:45,082 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:41<00:00,  5.11it/s]
2025-06-16 07:02:26,758 - easytorch-training - INFO - Result <val>: [val/time: 41.68 (s), val/loss: 2.6796, val/MAE: 2.6796, val/MAPE: 7.3576, val/RMSE: 5.1139]
100%|█████████████████████████████████████████| 427/427 [01:22<00:00,  5.15it/s]
2025-06-16 07:03:49,669 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6308, Test MAPE: 6.8373, Test RMSE: 5.0603
2025-06-16 07:03:49,671 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9783, Test MAPE: 8.2084, Test RMSE: 6.0759
2025-06-16 07:03:49,673 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3655, Test MAPE: 9.7979, Test RMSE: 7.1149
2025-06-16 07:03:49,695 - easytorch-training - INFO - Result <test>: [test/time: 82.94 (s), test/loss: 2.7694, test/MAE: 2.9382, test/MAPE: 8.0800, test/RMSE: 6.0207]
2025-06-16 07:03:50,876 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_092.pt saved
2025-06-16 07:03:50,877 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:09:22
2025-06-16 07:03:50,877 - easytorch-training - INFO - Epoch 93 / 100
100%|███████████████████████████████████████| 1498/1498 [06:00<00:00,  4.16it/s]
2025-06-16 07:09:51,183 - easytorch-training - INFO - Result <train>: [train/time: 360.31 (s), train/lr: 1.00e-05, train/loss: 2.5772, train/MAE: 2.5772, train/MAPE: 6.7220, train/RMSE: 5.0501]
2025-06-16 07:09:51,185 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:41<00:00,  5.11it/s]
2025-06-16 07:10:32,866 - easytorch-training - INFO - Result <val>: [val/time: 41.68 (s), val/loss: 2.6759, val/MAE: 2.6759, val/MAPE: 7.3016, val/RMSE: 5.1043]
100%|█████████████████████████████████████████| 427/427 [01:22<00:00,  5.15it/s]
2025-06-16 07:11:55,818 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6262, Test MAPE: 6.7879, Test RMSE: 5.0463
2025-06-16 07:11:55,820 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9737, Test MAPE: 8.1379, Test RMSE: 6.0592
2025-06-16 07:11:55,822 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3613, Test MAPE: 9.7195, Test RMSE: 7.0981
2025-06-16 07:11:55,845 - easytorch-training - INFO - Result <test>: [test/time: 82.98 (s), test/loss: 2.7653, test/MAE: 2.9344, test/MAPE: 8.0167, test/RMSE: 6.0065]
2025-06-16 07:11:57,026 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_093.pt saved
2025-06-16 07:11:57,026 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:09:17
2025-06-16 07:11:57,026 - easytorch-training - INFO - Epoch 94 / 100
100%|███████████████████████████████████████| 1498/1498 [06:00<00:00,  4.16it/s]
2025-06-16 07:17:57,397 - easytorch-training - INFO - Result <train>: [train/time: 360.37 (s), train/lr: 1.00e-05, train/loss: 2.5764, train/MAE: 2.5764, train/MAPE: 6.7196, train/RMSE: 5.0467]
2025-06-16 07:17:57,398 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:41<00:00,  5.11it/s]
2025-06-16 07:18:39,076 - easytorch-training - INFO - Result <val>: [val/time: 41.68 (s), val/loss: 2.6794, val/MAE: 2.6794, val/MAPE: 7.3292, val/RMSE: 5.1120]
100%|█████████████████████████████████████████| 427/427 [01:22<00:00,  5.15it/s]
2025-06-16 07:20:02,038 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6308, Test MAPE: 6.8115, Test RMSE: 5.0572
2025-06-16 07:20:02,040 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9795, Test MAPE: 8.1808, Test RMSE: 6.0748
2025-06-16 07:20:02,042 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3669, Test MAPE: 9.7787, Test RMSE: 7.1155
2025-06-16 07:20:02,064 - easytorch-training - INFO - Result <test>: [test/time: 82.99 (s), test/loss: 2.7703, test/MAE: 2.9395, test/MAPE: 8.0548, test/RMSE: 6.0208]
2025-06-16 07:20:03,228 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_094.pt saved
2025-06-16 07:20:03,228 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:09:11
2025-06-16 07:20:03,228 - easytorch-training - INFO - Epoch 95 / 100
100%|███████████████████████████████████████| 1498/1498 [06:00<00:00,  4.16it/s]
2025-06-16 07:26:03,356 - easytorch-training - INFO - Result <train>: [train/time: 360.13 (s), train/lr: 1.00e-05, train/loss: 2.5763, train/MAE: 2.5763, train/MAPE: 6.7213, train/RMSE: 5.0481]
2025-06-16 07:26:03,357 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:41<00:00,  5.11it/s]
2025-06-16 07:26:45,002 - easytorch-training - INFO - Result <val>: [val/time: 41.64 (s), val/loss: 2.6791, val/MAE: 2.6791, val/MAPE: 7.3519, val/RMSE: 5.1120]
100%|█████████████████████████████████████████| 427/427 [01:22<00:00,  5.15it/s]
2025-06-16 07:28:07,936 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6299, Test MAPE: 6.8291, Test RMSE: 5.0573
2025-06-16 07:28:07,938 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9795, Test MAPE: 8.1944, Test RMSE: 6.0782
2025-06-16 07:28:07,941 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3664, Test MAPE: 9.8130, Test RMSE: 7.1184
2025-06-16 07:28:07,962 - easytorch-training - INFO - Result <test>: [test/time: 82.96 (s), test/loss: 2.7699, test/MAE: 2.9389, test/MAPE: 8.0780, test/RMSE: 6.0229]
2025-06-16 07:28:09,121 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_095.pt saved
2025-06-16 07:28:09,121 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:09:05
2025-06-16 07:28:09,121 - easytorch-training - INFO - Epoch 96 / 100
100%|███████████████████████████████████████| 1498/1498 [06:00<00:00,  4.16it/s]
2025-06-16 07:34:09,496 - easytorch-training - INFO - Result <train>: [train/time: 360.37 (s), train/lr: 1.00e-05, train/loss: 2.5762, train/MAE: 2.5762, train/MAPE: 6.7188, train/RMSE: 5.0488]
2025-06-16 07:34:09,497 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:41<00:00,  5.11it/s]
2025-06-16 07:34:51,189 - easytorch-training - INFO - Result <val>: [val/time: 41.69 (s), val/loss: 2.6766, val/MAE: 2.6766, val/MAPE: 7.3336, val/RMSE: 5.1045]
100%|█████████████████████████████████████████| 427/427 [01:22<00:00,  5.15it/s]
2025-06-16 07:36:14,146 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6286, Test MAPE: 6.8053, Test RMSE: 5.0513
2025-06-16 07:36:14,148 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9768, Test MAPE: 8.1889, Test RMSE: 6.0688
2025-06-16 07:36:14,151 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3642, Test MAPE: 9.7839, Test RMSE: 7.1119
2025-06-16 07:36:14,174 - easytorch-training - INFO - Result <test>: [test/time: 82.98 (s), test/loss: 2.7678, test/MAE: 2.9370, test/MAPE: 8.0590, test/RMSE: 6.0156]
2025-06-16 07:36:15,341 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_096.pt saved
2025-06-16 07:36:15,342 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:09:00
2025-06-16 07:36:15,342 - easytorch-training - INFO - Epoch 97 / 100
100%|███████████████████████████████████████| 1498/1498 [06:00<00:00,  4.16it/s]
2025-06-16 07:42:15,841 - easytorch-training - INFO - Result <train>: [train/time: 360.50 (s), train/lr: 1.00e-05, train/loss: 2.5763, train/MAE: 2.5763, train/MAPE: 6.7188, train/RMSE: 5.0484]
2025-06-16 07:42:15,841 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:41<00:00,  5.11it/s]
2025-06-16 07:42:57,520 - easytorch-training - INFO - Result <val>: [val/time: 41.68 (s), val/loss: 2.6786, val/MAE: 2.6786, val/MAPE: 7.3542, val/RMSE: 5.1070]
100%|█████████████████████████████████████████| 427/427 [01:22<00:00,  5.15it/s]
2025-06-16 07:44:20,424 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6287, Test MAPE: 6.8333, Test RMSE: 5.0528
2025-06-16 07:44:20,426 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9766, Test MAPE: 8.2046, Test RMSE: 6.0661
2025-06-16 07:44:20,428 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3658, Test MAPE: 9.8007, Test RMSE: 7.1117
2025-06-16 07:44:20,451 - easytorch-training - INFO - Result <test>: [test/time: 82.93 (s), test/loss: 2.7684, test/MAE: 2.9370, test/MAPE: 8.0784, test/RMSE: 6.0138]
2025-06-16 07:44:21,628 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_097.pt saved
2025-06-16 07:44:21,628 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:08:55
2025-06-16 07:44:21,628 - easytorch-training - INFO - Epoch 98 / 100
100%|███████████████████████████████████████| 1498/1498 [06:00<00:00,  4.16it/s]
2025-06-16 07:50:22,030 - easytorch-training - INFO - Result <train>: [train/time: 360.40 (s), train/lr: 1.00e-05, train/loss: 2.5758, train/MAE: 2.5758, train/MAPE: 6.7165, train/RMSE: 5.0463]
2025-06-16 07:50:22,032 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:41<00:00,  5.12it/s]
2025-06-16 07:51:03,670 - easytorch-training - INFO - Result <val>: [val/time: 41.64 (s), val/loss: 2.6803, val/MAE: 2.6803, val/MAPE: 7.3419, val/RMSE: 5.1143]
100%|█████████████████████████████████████████| 427/427 [01:22<00:00,  5.15it/s]
2025-06-16 07:52:26,585 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6295, Test MAPE: 6.8226, Test RMSE: 5.0537
2025-06-16 07:52:26,588 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9781, Test MAPE: 8.1753, Test RMSE: 6.0700
2025-06-16 07:52:26,590 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3662, Test MAPE: 9.7687, Test RMSE: 7.1145
2025-06-16 07:52:26,612 - easytorch-training - INFO - Result <test>: [test/time: 82.94 (s), test/loss: 2.7690, test/MAE: 2.9384, test/MAPE: 8.0531, test/RMSE: 6.0180]
2025-06-16 07:52:27,786 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_098.pt saved
2025-06-16 07:52:27,786 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:08:50
2025-06-16 07:52:27,786 - easytorch-training - INFO - Epoch 99 / 100
100%|███████████████████████████████████████| 1498/1498 [06:00<00:00,  4.16it/s]
2025-06-16 07:58:28,038 - easytorch-training - INFO - Result <train>: [train/time: 360.25 (s), train/lr: 1.00e-05, train/loss: 2.5755, train/MAE: 2.5755, train/MAPE: 6.7164, train/RMSE: 5.0432]
2025-06-16 07:58:28,039 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:41<00:00,  5.11it/s]
2025-06-16 07:59:09,697 - easytorch-training - INFO - Result <val>: [val/time: 41.66 (s), val/loss: 2.6781, val/MAE: 2.6781, val/MAPE: 7.3288, val/RMSE: 5.1026]
100%|█████████████████████████████████████████| 427/427 [01:22<00:00,  5.15it/s]
2025-06-16 08:00:32,582 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6268, Test MAPE: 6.8054, Test RMSE: 5.0417
2025-06-16 08:00:32,584 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9755, Test MAPE: 8.1787, Test RMSE: 6.0575
2025-06-16 08:00:32,586 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3662, Test MAPE: 9.7804, Test RMSE: 7.1087
2025-06-16 08:00:32,609 - easytorch-training - INFO - Result <test>: [test/time: 82.91 (s), test/loss: 2.7672, test/MAE: 2.9364, test/MAPE: 8.0524, test/RMSE: 6.0080]
2025-06-16 08:00:33,742 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_099.pt saved
2025-06-16 08:00:33,742 - easytorch-training - INFO - The estimated training finish time is 2025-06-16 08:08:44
2025-06-16 08:00:33,742 - easytorch-training - INFO - Epoch 100 / 100
100%|███████████████████████████████████████| 1498/1498 [05:59<00:00,  4.16it/s]
2025-06-16 08:06:33,741 - easytorch-training - INFO - Result <train>: [train/time: 360.00 (s), train/lr: 1.00e-05, train/loss: 2.5748, train/MAE: 2.5748, train/MAPE: 6.7146, train/RMSE: 5.0455]
2025-06-16 08:06:33,743 - easytorch-training - INFO - Start validation.
100%|█████████████████████████████████████████| 213/213 [00:41<00:00,  5.13it/s]
2025-06-16 08:07:15,301 - easytorch-training - INFO - Result <val>: [val/time: 41.56 (s), val/loss: 2.6796, val/MAE: 2.6796, val/MAPE: 7.3625, val/RMSE: 5.1095]
100%|█████████████████████████████████████████| 427/427 [01:22<00:00,  5.16it/s]
2025-06-16 08:08:38,004 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6292, Test MAPE: 6.8154, Test RMSE: 5.0485
2025-06-16 08:08:38,006 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9785, Test MAPE: 8.2059, Test RMSE: 6.0688
2025-06-16 08:08:38,008 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3680, Test MAPE: 9.8284, Test RMSE: 7.1173
2025-06-16 08:08:38,031 - easytorch-training - INFO - Result <test>: [test/time: 82.73 (s), test/loss: 2.7697, test/MAE: 2.9390, test/MAPE: 8.0857, test/RMSE: 6.0177]
2025-06-16 08:08:39,181 - easytorch-training - INFO - Checkpoint checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_100.pt saved
2025-06-16 08:08:40,509 - easytorch-training - INFO - The training finished at 2025-06-16 08:08:40
2025-06-16 08:08:40,509 - easytorch-training - INFO - Evaluating the best model on the test set.
2025-06-16 08:08:40,509 - easytorch-training - INFO - Loading Checkpoint from 'checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/GraphLLM_best_val_MAE.pt'
100%|█████████████████████████████████████████| 427/427 [01:22<00:00,  5.18it/s]
2025-06-16 08:10:19,483 - easytorch-training - INFO - Evaluate best model on test data for horizon 3, Test MAE: 2.6179, Test MAPE: 6.7740, Test RMSE: 5.0166
2025-06-16 08:10:19,485 - easytorch-training - INFO - Evaluate best model on test data for horizon 6, Test MAE: 2.9518, Test MAPE: 8.0551, Test RMSE: 5.9820
2025-06-16 08:10:19,487 - easytorch-training - INFO - Evaluate best model on test data for horizon 12, Test MAE: 3.3345, Test MAPE: 9.5888, Test RMSE: 7.0154
2025-06-16 08:10:19,710 - easytorch-training - INFO - Result <test>: [test/time: 82.70 (s), test/loss: 2.7478, test/MAE: 2.9157, test/MAPE: 7.9504, test/RMSE: 5.9386]
2025-06-16 08:10:19,711 - easytorch-training - INFO - Test results saved to checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/test_results.npz.
2025-06-16 08:10:19,711 - easytorch-training - INFO - Test metrics saved to checkpoints/GraphLLM/METR-LA_100_12_12/edce10106d3be141f46150cbb7e813ab/test_metrics.json.

Process finished with exit code 0
'''