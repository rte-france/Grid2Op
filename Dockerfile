# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

# Use an official Python runtime as a parent image
FROM ubuntu:18.04

MAINTAINER Benjamin DONNOT <benjamin.donnot@rte-france.com>

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && \
    apt-get install -y \
    less \
    apt-transport-https \
    apt-utils \
    git \
    ssh \
    tar \
    gzip \
    ca-certificates \
    python3 \
    python3-pip \
    software-properties-common  # for add-apt-repository command

# install julia 1.4
RUN add-apt-repository ppa:jonathonf/julialang
RUN apt-get update
RUN apt-get install -y julia

# weird 'hack' for julia in some ubuntu
RUN ln /usr/lib/x86_64-linux-gnu/libjulia.so.1 /usr/lib/x86_64-linux-gnu/libjulia.so

# install python in julia
RUN julia -e 'ENV["PYTHON"]="/usr/bin/python3"; using Pkg;  Pkg.add("Ipopt"); Pkg.add("PowerModels"); Pkg.add("PyCall"); Pkg.add("Juniper"); Pkg.add("Cbc"); Pkg.add("JSON")'

# install julia from python
RUN pip3 install julia  # install julia in python

# Retrieve Grid2Op
RUN git clone https://github.com/rte-france/Grid2Op

# Install Grid2Op
WORKDIR /Grid2Op
# Use the latest release
RUN git pull
RUN git remote update
RUN git fetch --all --tags
RUN git checkout "tags/v0.8.1" -b "v0.8.1-branch"
# Install Dependencies
RUN pip3 install .  #[optional,challenge]
WORKDIR /

# Make port 80 available to the world outside this container
EXPOSE 80