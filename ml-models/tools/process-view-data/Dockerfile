FROM continuumio/miniconda3
RUN apt-get update && apt-get install -y libgl1
WORKDIR /app
COPY environment.yml .
RUN conda env create -f environment.yml
ENV PATH /opt/conda/envs/process-view-data/bin:$PATH
RUN /bin/bash -c "source activate process-view-data"
RUN pip install geopandas tqdm opencv-python pyshp environs
RUN pip install requests
COPY . .
CMD python src/main.py