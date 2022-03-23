# Copyright (c) 2019-2021, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.


class DoNothingLog:
    """
    INTERNAL

    .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

    A class to emulate the behaviour of a logger, but that does absolutely nothing.
    """

    INFO = 2
    WARNING = 1
    ERROR = 0

    def __init__(self, max_level=2):
        self.max_level = max_level

    def warn(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


class ConsoleLog(DoNothingLog):
    """
    INTERNAL

    .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

    A class to emulate the behaviour of a logger, but that prints on the console
    """

    def __init__(self, max_level=2):
        DoNothingLog.__init__(self, max_level)

    def warn(self, *args, **kwargs):
        if self.max_level >= self.WARNING:
            if args:
                print('WARNING: "{}"'.format(", ".join(args)))
            if kwargs:
                print("WARNING: {}".format(kwargs))

    def info(self, *args, **kwargs):
        if self.max_level >= self.INFO:
            if args:
                print('INFO: "{}"'.format(", ".join(args)))
            if kwargs:
                print("INFO: {}".format(kwargs))

    def error(self, *args, **kwargs):
        if self.max_level >= self.ERROR:
            if args:
                print('ERROR: "{}"'.format(", ".join(args)))
            if kwargs:
                print("ERROR: {}".format(kwargs))

    def warning(self, *args, **kwargs):
        if self.max_level >= self.WARNING:
            if args:
                print('WARNING: "{}"'.format(", ".join(args)))
            if kwargs:
                print("WARNING: {}".format(kwargs))
