# From base image: tf1 or tf2
FROM registry.eofactory.ai:5000/eof/base-images/pytorch1.8.2
# FROM pytorch1.8.2
# Install additional dependencies 
RUN pip install environs
RUN pip3 install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111
# RUN pip install torchvision
# RUN pip install something
# RUN conda install something
# RUN pip install geopandas scikit-learn scikit-image
# COPY source code to container

COPY . .
# Command to run 
CMD python src/main.py
