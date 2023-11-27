# ubuntu=20.04
# gdal=3.4.1
# pytorch=1.8.2
# opencv-python==4.7.0.68
# rasterio==1.3.4
# tqdm==4.64.1

#FROM base image ubuntu
FROM ubuntu:20.04

#Update and install some linux dependencies to use tednsorflow
RUN apt-get update && apt-get -y upgrade && apt-get install -y wget gcc 


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
ENV LD_LIBRARY_PATH /opt/conda/envs/pytorch_sam/lib

ENV PATH /opt/conda/envs/pytorch_sam/bin:$PATH

RUN /bin/bash -c "source activate pytorch_sam"


