# Copyright (c) 2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.


class HighResSimCounter:
    """This classes helps to count the total number of call to "high fidelity simulator"
    the agent made.
    """
    def __init__(self) -> None:
        self.__nb_highres_called = 0
        
    def __iadd__(self, other):
        self.__nb_highres_called += int(other)
        return self
        
    def add_one(self):
        self.__nb_highres_called += 1
    
    @property
    def nb_highres_called(self):
        return self.__nb_highres_called
    