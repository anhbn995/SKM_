# Don't change
from environs import Env
env = Env()
env.read_env()

# This is path where store input data. Input path is a directory that contains multi images or a image path
INPUT_IMAGES_DIR = env.str('INPUT_IMAGES_DIR')
INPUT_LABEL_PATH = env.str('INPUT_LABEL_PATH')
SAMPLE_SIZE = env.int('SAMPLE_SIZE', 0) or None
GEN_THEM = env.bool('GEN_THEM', False)
SPLIT = env.float('SPLIT', 0.8)
TMP_PATH = env.str('TMP_PATH')
OUTPUT_DIR = env.str('OUTPUT_DIR')

TASK_ID = env.int('TASK_ID', -1)

BROADCAST = env.str('BROADCAST', '')