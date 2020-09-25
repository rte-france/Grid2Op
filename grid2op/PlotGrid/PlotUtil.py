# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import numpy as np


class PlotUtil:
    """
    .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

    This is a wrapper that contains utilities to draw the information on the plots more easily.
    """
    @staticmethod
    def format_value_unit(value, unit):
        if isinstance(value, float):
            return "{:.2f} {}".format(value, unit)
        elif isinstance(value, int):
            return "{:d} {}".format(value, unit)
        else:
            return "{} {}".format(value, unit)

    @staticmethod
    def middle_from_points(x1, y1, x2, y2):
        return (x1 + x2) * 0.5, (y1 + y2) * 0.5

    @staticmethod
    def vec_from_points(x1, y1, x2, y2):
        return x2 - x1, y2 - y1

    @staticmethod
    def mag_from_points(x1, y1, x2, y2):
        x, y = PlotUtil.vec_from_points(x1, y1, x2, y2)
        return np.linalg.norm([x, y])

    @staticmethod
    def norm_from_points(x1, y1, x2, y2):
        x, y, = PlotUtil.vec_from_points(x1, y1, x2, y2)
        return PlotUtil.norm_from_vec(x, y)

    @staticmethod
    def norm_from_vec(x, y):
        n = np.linalg.norm([x, y])
        return x / n, y / n

    @staticmethod
    def orth_from_points(x1, y1, x2, y2):
        x, y = PlotUtil.vec_from_points(x1, y1, x2, y2)
        return -y, x

    @staticmethod
    def orth_norm_from_points(x1, y1, x2, y2):
        x, y = PlotUtil.vec_from_points(x1, y1, x2, y2)
        xn, yn = PlotUtil.norm_from_vec(x, y)
        return -yn, xn
