# From base image: tf1 or tf2
FROM registry.eofactory.ai:5000/eof/base-images/tf2.7
# FROM pytorch1.8.2
# Install additional dependencies 
# RUN pip install torchvision
# RUN pip install something
# RUN conda install something
RUN pip install geopandas scikit-learn scikit-image
# COPY source code to container

COPY . .
# Command to run 
CMD python src/main.py
