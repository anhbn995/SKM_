import subprocess
from subprocess import CalledProcessError
from params import ROOT_DATA_FOLDER
import os
import uuid
model_name = uuid.uuid4().hex
build_command = ['docker', 'build', '-t', model_name, '.']
run_command = ['docker', 'run', '--rm', '--gpus',
               'all', '-v', f'{ROOT_DATA_FOLDER}:{ROOT_DATA_FOLDER}', '-v', f'{os.getcwd()}/.env:/app/.env', model_name]

remove_command = ['docker', 'image', 'rm', '-f', model_name]
try:
    subprocess.run(build_command)
    subprocess.run(run_command)
except CalledProcessError as e:
    print(e.stderr)
finally:
    subprocess.run(remove_command)
