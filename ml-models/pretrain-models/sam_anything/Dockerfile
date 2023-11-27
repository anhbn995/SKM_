# From base image: tf1 or tf2
FROM registry.eofactory.ai:5000/eof/base-images/pytorch_sam:latest
# Install additional dependencies 
ENTRYPOINT []
RUN apt install -y libgl1
# RUN pip install something
# RUN conda install something
RUN pip install segment-geospatial==0.8.1 environs rasterio==1.3.7
# COPY source code to container
COPY . .
# Command to run 
CMD python src/main.py
