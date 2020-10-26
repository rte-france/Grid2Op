#/bin/bash

# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.


if [ $# -eq 0 ]
  then
    echo "No arguments supplied, please specified the grid2op version to push to docker"
    exit 1
fi
version=$1

echo "Pushing grid2ip verion "$version
#exit 1
docker build -t bdonnot/grid2op:$version .
docker push bdonnot/grid2op:$version
docker build -t bdonnot/grid2op:latest .
docker push bdonnot/grid2op:latest

