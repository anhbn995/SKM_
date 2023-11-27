# From base image: tf1 or tf2
FROM registry.eofactory.ai:5000/eof/base-images/tf1.15
# Install additional dependencies 

RUN pip install scikit-image==0.18.1 pyshp==2.1.0 rio-tiler==2.0.0b17
# RUN conda install something

# COPY source code to container
COPY . .
# Command to run 
CMD python src/main.py
