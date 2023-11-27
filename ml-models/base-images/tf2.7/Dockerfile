# ubuntu=20.04
# gdal=3.4.0
# tensorflow=2.7.0
# opencv-python=4.5.5.64
# tqdm=4.64.0
# rasterio==1.2.10

#FROM base image ubuntu
FROM ubuntu:20.04

#Update and install some linux dependencies to use tensorflow
RUN apt-get update && apt-get -y upgrade && apt-get install -y wget ffmpeg libsm6 libxext6


#Install miniconda
ENV CONDA_DIR /opt/conda

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
  /bin/bash ~/miniconda.sh -b -p /opt/conda

ENV PATH=$CONDA_DIR/bin:$PATH

#Install packages from environment file
WORKDIR /app

COPY environment.yaml environment.yaml

RUN conda env create -f environment.yaml

ENV PYTHONPATH "${PYTHONPATH}:/app"

# Set LD_LIBRARY_PATH and activate env
ENV LD_LIBRARY_PATH /opt/conda/envs/tf2.7/lib

ENV PATH /opt/conda/envs/tf2.7/bin:$PATH

RUN /bin/bash -c "source activate tf2.7"


