# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import unittest
import tempfile
import json

from grid2op.Parameters import Parameters


class TestParameters(unittest.TestCase):
    def test_default_builds(self):
        p = Parameters()

    def test_to_dict(self):
        p = Parameters()
        p_dict = p.to_dict()
        assert isinstance(p_dict, dict)

    def test_init_from_dict(self):
        p = Parameters()
        p_dict = p.to_dict()

        p_dict["NB_TIMESTEP_OVERFLOW_ALLOWED"] = 42

        p.init_from_dict(p_dict)
        assert p.NB_TIMESTEP_OVERFLOW_ALLOWED == 42

    def test_from_json(self):
        p = Parameters()
        p_dict = p.to_dict()
        p_dict["NB_TIMESTEP_OVERFLOW_ALLOWED"] = 42
        p_json = json.dumps(p_dict, indent=2)
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf = tempfile.NamedTemporaryFile(delete=False)
            tf.write(bytes(p_json, "utf-8"))
            tf.close()

            p2 = Parameters.from_json(tf.name)
            assert p2.NB_TIMESTEP_OVERFLOW_ALLOWED == 42

    def test_init_from_json(self):
        p = Parameters()
        p_dict = p.to_dict()
        p_dict["NB_TIMESTEP_OVERFLOW_ALLOWED"] = 42
        p_json = json.dumps(p_dict, indent=2)
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(bytes(p_json, "utf-8"))
            tf.close()

            p.init_from_json(tf.name)
            assert p.NB_TIMESTEP_OVERFLOW_ALLOWED == 42

    def _aux_check_attr_int(self, param, attr_name):
        # don't work with string
        setattr(param, attr_name, "toto")
        with self.assertRaises(RuntimeError):
            param.check_valid()
        # don't work with list
        setattr(param, attr_name, [1, 2])
        with self.assertRaises(RuntimeError):
            param.check_valid()
        # dont work with set
        setattr(param, attr_name, {1, 2})
        with self.assertRaises(RuntimeError):
            param.check_valid()
        # work with bool
        setattr(param, attr_name, True)
        param.check_valid()
        # work with float
        setattr(param, attr_name, 2.0)
        param.check_valid()
        # work with int
        setattr(param, attr_name, 1)
        param.check_valid()
        # check fail negative
        setattr(param, attr_name, -2)
        with self.assertRaises(RuntimeError):
            param.check_valid()

        # restore correct value
        setattr(param, attr_name, 1)
        param.check_valid()

    def _aux_check_attr_float(self, param, attr_name, test_bool=True):
        tmp = getattr(param, attr_name)
        # don't work with string
        setattr(param, attr_name, "toto")
        with self.assertRaises(RuntimeError):
            param.check_valid()
        # don't work with list
        setattr(param, attr_name, [1, 2])
        with self.assertRaises(RuntimeError):
            param.check_valid()
        # dont work with set
        setattr(param, attr_name, {1, 2})
        with self.assertRaises(RuntimeError):
            param.check_valid()
        # work with bool
        if test_bool:
            setattr(param, attr_name, True)
            param.check_valid()
        # work with int value
        setattr(param, attr_name, int(tmp))
        param.check_valid()
        # work with init value
        setattr(param, attr_name, tmp)
        param.check_valid()

    def test_check_valid(self):
        """test the param.check_valid() is working correctly (accept valid param, reject wrong param)"""
        p = Parameters()
        p.check_valid()  # default params are valid

        # boolean attribute
        p.NO_OVERFLOW_DISCONNECTION = 1
        with self.assertRaises(RuntimeError):
            p.check_valid()
        p.NO_OVERFLOW_DISCONNECTION = True
        p.IGNORE_MIN_UP_DOWN_TIME = "True"
        with self.assertRaises(RuntimeError):
            p.check_valid()
        p.IGNORE_MIN_UP_DOWN_TIME = True
        p.ACTIVATE_STORAGE_LOSS = 42.0
        with self.assertRaises(RuntimeError):
            p.check_valid()
        p.ACTIVATE_STORAGE_LOSS = False
        p.ENV_DC = [1, 2]
        with self.assertRaises(RuntimeError):
            p.check_valid()
        p.ENV_DC = True
        p.ALLOW_DISPATCH_GEN_SWITCH_OFF = {1, 2}
        with self.assertRaises(RuntimeError):
            p.check_valid()
        p.ALLOW_DISPATCH_GEN_SWITCH_OFF = False
        p.check_valid()  # everything valid again

        # int types
        for attr_nm in [
            "NB_TIMESTEP_OVERFLOW_ALLOWED",
            "NB_TIMESTEP_RECONNECTION",
            "NB_TIMESTEP_COOLDOWN_LINE",
            "NB_TIMESTEP_COOLDOWN_SUB",
            "MAX_SUB_CHANGED",
            "MAX_LINE_STATUS_CHANGED",
        ]:
            try:
                self._aux_check_attr_int(p, attr_nm)
            except Exception as exc_:
                raise RuntimeError(f'Exception "{exc_}" for attribute "{attr_nm}"')
        # float types
        for attr_nm in ["HARD_OVERFLOW_THRESHOLD",
                        "SOFT_OVERFLOW_THRESHOLD",
                        "INIT_STORAGE_CAPACITY"]:
            try:
                self._aux_check_attr_float(p, attr_nm, test_bool=(attr_nm!="HARD_OVERFLOW_THRESHOLD"))
            except Exception as exc_:
                raise RuntimeError(f'Exception "{exc_}" for attribute "{attr_nm}"')

        p.HARD_OVERFLOW_THRESHOLD = 0.5  # should not validate
        with self.assertRaises(RuntimeError):
            p.check_valid()
        p.ACTIVATE_STORAGE_LOSS = -0.1  # should not validate
        with self.assertRaises(RuntimeError):
            p.check_valid()
        p.ACTIVATE_STORAGE_LOSS = 1.1  # should not validate
        with self.assertRaises(RuntimeError):
            p.check_valid()


if __name__ == "__main__":
    unittest.main()
