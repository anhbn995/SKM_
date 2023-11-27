# From base image: tf1 or tf2
FROM registry.eofactory.ai:5000/eof/base-images/tf1.15
# Install additional dependencies 

# RUN pip install something
RUN pip install geopandas scikit-image
# RUN conda install something

# COPY source code to container
COPY . .
# Command to run 
CMD python -m src.main
