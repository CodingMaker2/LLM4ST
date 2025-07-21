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
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

def parse_args():
    parser = ArgumentParser(description='Run time series forecasting model in BasicTS framework!')
    parser.add_argument('-c', '--cfg', default='baselines/GraphPoolLLM/PEMS08_copy4.py', help='training config')
    parser.add_argument('-g', '--gpus', default='2', help='visible gpus')
    return parser.parse_args()


def main():
    args = parse_args()
    basicts.launch_training(args.cfg, args.gpus, node_rank=0)


if __name__ == '__main__':
    main()

