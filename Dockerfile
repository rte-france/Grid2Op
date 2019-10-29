# Use an official Python runtime as a parent image
FROM python:3.6-stretch

MAINTAINER Benjamin DONNOT <benjamin.donnot@rte-france.com>

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && \
    apt-get install -y \
    less \
    apt-transport-https \
    software-properties-common

# Install octave
RUN apt-get remove -y software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Retrieve Grid2Op
RUN pip3 install -U numba matplotlib scipy numpy pandas pyaml pygame
RUN git clone https://github.com/rte-france/Grid2Op

# Install Grid2Op (including necessary packages installation)
WORKDIR Grid2Op/
RUN cd /Grid2Op && pip install -U . && cd ..

# Make port 80 available to the world outside this container
EXPOSE 80

# Run the sample experiments when the container launches
CMD ["python3.6", "-m", "grid2op.main"]