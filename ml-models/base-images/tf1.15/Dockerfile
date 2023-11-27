# ubuntu=20.04
# gdal=2.3.2
# tensorflow=1.15.0
# opencv-python=4.5.5.64
# tqdm=4.63.1
# rasterio==1.1.7
# rio_tiler=3.1.4
# shapely=1.6.4
# geopandas=0.10.2

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

COPY environment.yaml environment.tf1.15.yaml

RUN conda env create -f environment.tf1.15.yaml

ENV PYTHONPATH "${PYTHONPATH}:/app"

ENV PATH /opt/conda/envs/tf1.15/bin:$PATH

RUN /bin/bash -c "source activate tf1.15"


