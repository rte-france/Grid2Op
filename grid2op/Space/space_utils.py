# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from grid2op.Exceptions import Grid2OpException


def extract_from_dict(dict_, key, converter):
    if not key in dict_:
        raise Grid2OpException("Impossible to find key \"{}\" while loading the dictionnary.".format(key))
    try:
        res = converter(dict_[key])
    except:
        raise Grid2OpException("Impossible to convert \"{}\" into class {}".format(key, converter))
    return res


def save_to_dict(res_dict, me, key, converter):
    if not hasattr(me, key):
        raise Grid2OpException("Impossible to find key \"{}\" while loading the dictionnary.".format(key))
    try:
        res = converter(getattr(me, key))
    except:
        raise Grid2OpException("Impossible to convert \"{}\" into class {}".format(key, converter))

    if key in res_dict:
        msg_err_ = "Key \"{}\" is already present in the result dictionnary. This would override it" \
                   " and is not supported."
        raise Grid2OpException(msg_err_.format(key))
    res_dict[key] = res
