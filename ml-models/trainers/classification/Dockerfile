# From base image: tf1 or tf2
FROM registry.eofactory.ai:5000/eof/base-images/tf1.15
# Install additional dependencies 

# COPY source code to container
RUN pip install scikit-learn requests geopy rio-tiler==2.0.0b17

COPY . .
# Command to run 
CMD python src/main.py
