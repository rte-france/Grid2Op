# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

# Use an official Python runtime as a parent image
FROM python:3.8-buster

MAINTAINER Benjamin DONNOT <benjamin.donnot@rte-france.com>

ENV DEBIAN_FRONTEND noninteractive

ARG ls_version

# generic install
RUN apt-get update && \
    apt-get install -y \
    less \
    apt-transport-https \
    build-essential \
    git \
    ssh \
    tar \
    gzip

# Retrieve Grid2Op
RUN git clone https://github.com/rte-france/Grid2Op

# Install Grid2Op
WORKDIR /Grid2Op
# Use the latest release
RUN git pull
RUN git remote update
RUN git fetch --all --tags
RUN git checkout "tags/v1.10.1" -b "v1.10.1-branch"
# Install Dependencies
RUN pip3 install .[optional,challenge]
WORKDIR /

# Make port 80 available to the world outside this container
EXPOSE 80