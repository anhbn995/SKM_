FROM mundialis/esa-snap:9.0-ubuntu
RUN apt-get update -y && apt-get install snaphu -y
RUN pip install environs
WORKDIR /app
COPY . .

ENTRYPOINT [ "/bin/bash", "-l", "-c" ]
CMD ["python3 gen_DEM_sen1.py"]