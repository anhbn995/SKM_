# From base image: tf1 or tf2
FROM registry.eofactory.ai:5000/eof/base-images/tf2.7
RUN apt update && apt-get install -y gcc
RUN pip install scipy scikit-image scikit-image requests 
RUN conda install keras
RUN pip install pycocotools
# Install additional dependencies 
COPY . .
# Command to run 
CMD python src/training_treecouting.py

