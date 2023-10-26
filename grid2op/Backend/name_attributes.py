# Copyright (c) 2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import numpy as np

def _line_name(row, id_obj):
    return f"{row['from_bus']}_{row['to_bus']}_{id_obj}"


def _trafo_name(row, id_obj):
    return f"{row['hv_bus']}_{row['lv_bus']}_{id_obj}"


def _gen_name(row, id_obj):
    return f"gen_{row['bus']}_{id_obj}"


def _load_name(row, id_obj):
    return f"load_{row['bus']}_{id_obj}"


def _storage_name(row, id_obj):
    return f"storage_{row['bus']}_{id_obj}"


def _sub_name(row, id_obj):
    return f"sub_{id_obj}"


def _shunt_name(row, id_obj):
    return f"shunt_{row['bus']}_{id_obj}"


def _aux_get_names(grid, grid_attrs):
    res = []
    obj_id = 0
    for (attr, fun_to_name) in grid_attrs:
        df = getattr(grid, attr)
        if (
            "name" in df.columns
            and not df["name"].isnull().values.any()
        ):
            res += [name for name in df["name"]]
        else:
            res += [
                fun_to_name(row, id_obj=obj_id + i)
                for i, (_, row) in enumerate(df.iterrows())
            ]
            obj_id += df.shape[0]
    res = np.array(res)
    return res


def get_pandapower_default_names(pp_grid):
    sub_names = ["sub_{}".format(i) for i, row in pp_grid.bus.iterrows()]
    load_names = _aux_get_names(pp_grid, [("load", _load_name)])
    shunt_names = _aux_get_names(pp_grid, [("shunt", _shunt_name)])
    gen_names = _aux_get_names(pp_grid, [("gen", _gen_name), ("ext_grid", _gen_name)])
    line_names = _aux_get_names(pp_grid, [("line", _line_name), ("trafo", _trafo_name)])
    storage_names = _aux_get_names(pp_grid, [("storage", _storage_name)])
    return sub_names, load_names, shunt_names, gen_names, line_names, storage_names
