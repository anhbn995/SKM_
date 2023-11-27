# From base image: tf1 or tf2
FROM registry.eofactory.ai:5000/pretrain-models/sam_text:1.0.2
# Install additional dependencies 
ENTRYPOINT []
RUN apt install -y libgl1
# RUN pip install something
# RUN conda install something
RUN pip install segment-geospatial environs flask-restful
# COPY source code to container
COPY . .
# Command to run 
CMD python src/main.py

