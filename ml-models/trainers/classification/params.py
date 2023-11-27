from environs import Env
env = Env()
env.read_env()

ROOT_DATA_FOLDER = env.str('ROOT_DATA_FOLDER')

IMAGE_PATH = env.str('IMAGE_PATH')

OUTPUT_DIR = env.str('OUTPUT_DIR')

TMP_PATH = env.str('TMP_PATH')

TASK_ID = env.int('TASK_ID', -1)

BROADCAST = env.str('BROADCAST', '')

HOSTED_ENDPOINT = env.str('HOSTED_ENDPOINT', '')

BOUND_PATH = env.str('BOUND_PATH')

ANNOTATION_PATH = env.str('ANNOTATION_PATH')

LABELS = env.str('LABELS', '[]')

INPUT_TYPE = env.str('INPUT_TYPE', 'vector')

TRAINER_TYPE = env.str('TRAINER_TYPE', 'unet')

MASK = env.str('MASK', '[]')

TRAINER_SIZE = env.int('TRAINER_SIZE', 128)
OPTIMIZER = env.str('OPTIMIZER', 'adam')
LOSS = env.str('LOSS', 'categorical_crossentropy')
METRICS = env.str('METRICS', '["accuracy"]')
EPOCHS = env.int('EPOCHS', 50)
N_FILTERS = env.int('N_FILTERS', 16)
DROPOUT = env.float('DROPOUT', 0.5)
BATCHNORM = env.bool('BATCHNORM', True)
