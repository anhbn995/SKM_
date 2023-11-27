from environs import Env
env = Env()
env.read_env()

ROOT_DATA_FOLDER = env.str('ROOT_DATA_FOLDER')

IMAGE_PATH = env.str('IMAGE_PATH')

OUTPUT_DIR = env.str('OUTPUT_DIR')

TMP_PATH = env.str('TMP_PATH')

TASK_ID = env.int('TASK_ID', -1)

HOSTED_ENDPOINT = env.str('HOSTED_ENDPOINT', '')

BOUND_PATH = env.str('BOUND_PATH')

ANNOTATION_PATH = env.str('ANNOTATION_PATH')

EPOCHS = env.int('EPOCHS', 100)
