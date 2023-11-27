# Don't change
from environs import Env
env = Env()
env.read_env()

# This is path where store input data. Input path is a directory that contains multi images or a image path
INPUT_PATH_1 = env.str('INPUT_PATH_1')
INPUT_PATH_2 = env.str('INPUT_PATH_2')
FIRST_BURST = env.int('FIRST_BURST',1)
LAST_BURST = env.int('LAST_BURST',2)
SWATH = env.int('SWATH','IW2')
TMP_PATH = env.str('TMP_PATH')
OUTPUT_PATH = env.str('OUTPUT_PATH')

