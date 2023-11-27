from environs import Env
env = Env()
env.read_env()
ROOT_DATA_FOLDER = env.str('ROOT_DATA_FOLDER')
INPUT_PATH = env.str('INPUT_PATH')
OUTPUT_PATH = env.str('OUTPUT_PATH')
TMP_PATH = env.str('TMP_PATH')
DIL_RESULT = env.bool('DIL_RESULT', True)
RUN_AGAIN = env.bool('RUN_AGAIN', True)
THRESH_HOLD_GREEN = env.float('THRESH_HOLD_GREEN', 0.15)
THRESH_HOLD_WATER = env.float('THRESH_HOLD_WATER', 0.15)
