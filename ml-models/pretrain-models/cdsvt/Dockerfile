# From base image: tf1 or tf2
FROM registry.eofactory.ai:5000/eof/base-images/tf2.7
# Install additional dependencies 

RUN pip install pandas pytesseract environs matplotlib pdf2image geopandas scipy scikit-image shapely==1.8.0
# RUN pip install pytesseract
# RUN pip install environs
# RUN pip install matplotlib
# RUN pip install shapely==1.8.0
# RUN pip install pdf2image
# RUN pip install geopandas
# RUN pip install scipy
# RUN pip install scikit-image
RUN apt-get install -y tesseract-ocr
# COPY source code to container
COPY . .
# COPY /path/to/additional/folder /path/inside/container
# Command to run 
CMD python src/main.py
