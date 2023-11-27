# Don't change
from environs import Env
env = Env()
env.read_env()
# This is path where store input data. Input path is a directory that contains multi images or a image path
TRAINING_DATASET_DIRS = env.list('TRAINING_DATASET_DIRS')
OUTPUT_DIR = env.str('OUTPUT_DIR')
