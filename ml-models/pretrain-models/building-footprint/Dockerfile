# From base image: tf1 or tf2
FROM registry.eofactory.ai:5000/eof/base-images/tf1.15
# Install additional dependencies 

# RUN pip install something
# RUN conda install something

# COPY source code to container
RUN pip install scikit-image rtree
COPY . .
# Command to run 
CMD python src/main.py
