# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

"""
In this module are defined all the exceptions that are used in the Grid2Op package.

They all inherit from :class:`Grid2OpException`, which is nothing more than a :class:`RuntimeError` with
customs :func:`Grid2OpException.__repr__` and :func:`Grid2OpException.__str__` definition to allow for easier logging.

"""
import inspect


class Grid2OpException(RuntimeError):
    """
    Base Exception from which all Grid2Op raise exception derived.
    """
    def vect_hierarchy_cleaned(self):
        hierarchy = inspect.getmro(self.__class__)
        names_hierarchy = [el.__name__ for el in hierarchy]
        names_hierarchy = names_hierarchy[::-1]
        # i = names_hierarchy.index("RuntimeError")
        i = names_hierarchy.index("Grid2OpException")
        names_hierarchy = names_hierarchy[i:]
        res = " ".join(names_hierarchy) + " "
        return res

    def __repr__(self):
        res = self.vect_hierarchy_cleaned()
        res += RuntimeError.__repr__(self)
        return res

    def __str__(self):
        res = self.vect_hierarchy_cleaned()
        res += "\"{}\"".format(RuntimeError.__str__(self))
        return res
