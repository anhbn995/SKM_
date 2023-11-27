# From base image: tf1 or tf2
FROM registry.eofactory.ai:5000/eof/base-images/tf2.7
# Install additional dependencies 

RUN pip install scikit-image==0.18.1 pyshp==2.1.0 rio-tiler==2.0.0b17 pandas==2.0.1 rtree==1.0.1 geopandas==0.13.0 shapely==1.8.4
# RUN conda install something

# COPY source code to container
COPY . .
# Command to run 
CMD python src/main.py
# CMD python src/predict_treecounting.py
