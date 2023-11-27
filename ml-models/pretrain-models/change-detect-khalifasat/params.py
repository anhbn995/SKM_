# Don't change
from environs import Env
env = Env()
env.read_env()

# YOUR PARAMETER HERE

# Please make sure every model must have 4 input below
# This is root folder where related data stored. You must name your folder in root folder as same as your model folder name
# Example: ROOT_DATA_FOLDER/type-of-model/model-name
ROOT_DATA_FOLDER = env.str('ROOT_DATA_FOLDER')

# This is path where store input data. Input path is a directory that contains multi images or a image path
INPUT_PATH_1 = env.str('INPUT_PATH_1')
INPUT_PATH_2 = env.str('INPUT_PATH_2')
# This is path where store output data. Input path is a directory that contains multi images or a image path or a
# You have to write your output to output path
OUTPUT_PATH = env.str('OUTPUT_PATH')

# This is tmp folder where store tmp data when running your model
TMP_PATH = env.str('TMP_PATH')


# MODEL_PATH = env.str('MODEL_PATH')



# This is another input base on your model. For convenient, you must declare the type of the input.
# If the input is not required please add a default value
# THRESHOLD = env.float('THRESHOLD', 0.8)
