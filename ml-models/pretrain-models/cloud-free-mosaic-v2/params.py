from environs import Env
env = Env()
env.read_env()
ROOT_DATA_FOLDER = env.str('ROOT_DATA_FOLDER')
INPUT_PATH = env.str('INPUT_PATH')
OUTPUT_PATH = env.str('OUTPUT_PATH')
TMP_PATH = env.str('TMP_PATH')