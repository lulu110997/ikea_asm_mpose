# GENERAL LIBRARIES
# https://www.tensorflow.org/install/source#gpu
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import argparse
from datetime import datetime
# MACHINE LEARNING LIBRARIES
import numpy as np
import tensorflow as tf
# CUSTOM LIBRARIES
from AcT_utils.tools import read_yaml, Logger
from AcT_utils.trainer import Trainer

MODEL_SZ = "micro"

# LOAD CONFIG 
parser = argparse.ArgumentParser(description='Process some input')
parser.add_argument('--config', default='AcT_utils/config.yaml', type=str, help='Config path', required=False)
parser.add_argument('--test', '-t', action='store_true', help='Run a test (specify weights in the config file)') 
parser.add_argument('--benchmark','-b', action='store_true', help='Run a benchmark') 
parser.add_argument('--search','-s', action='store_true', help='Run a random search')
args = parser.parse_args()

config = read_yaml(args.config)

# Save dirs for results, bin and logs
now = datetime.now()
now = now.strftime("%y%m%d%H%M%S")
results_dirs = []
for entry in ['MODEL_DIR','RESULTS_DIR','LOG_DIR']:
    if not os.path.exists(config[entry]):
        results_dirs.append(os.path.join(now, config[entry]))
        os.makedirs(os.path.join(now, config[entry]))


# SET GPU 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) == 0:
    raise "gpu not found"
tf.config.experimental.set_visible_devices(gpus[config['GPU']], 'GPU')
tf.config.experimental.set_memory_growth(gpus[config['GPU']], True)


logger = Logger(results_dirs[2] + f'/{MODEL_SZ}_log.txt')
# SET TRAINER
trainer = Trainer(config, logger, MODEL_SZ, results_dirs)

if args.benchmark:
    # RUN BENCHMARK
    trainer.do_benchmark()

elif True or args.search:
    # RUN RANDOM SEARCH
    try:
        trainer.do_random_search()
    except Exception as e:
        raise e

elif args.test:
    # RUN TEST
    trainer.do_test()

else:
    print('Nothing to do! Specify one of the following arguments:')
    print('\t --benchmark [-b]: run a benchmark')
    print('\t --search [-s]: run a random search')
    print('\t --test [-t]: run a test')