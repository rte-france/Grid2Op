# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

# do some generic tests that can be implemented directly to test if a backend implementation can work out of the box
# with grid2op.
# see an example of test_Pandapower for how to use this suit.
import unittest
import numpy as np
import warnings

import grid2op
from grid2op.Backend.educPandaPowerBackend import EducPandaPowerBackend
from grid2op.Exceptions import EnvError


class TestAuxFunctions(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.envref = grid2op.make(
                "rte_case14_realistic",
                test=True,
                _add_to_name=type(self).__name__+"test_gridobjects_testauxfunctions",
            )
        seed = 0
        self.nb_test = 10
        self.max_iter = 30

        self.envref.seed(seed)
        self.seeds = [
            i for i in range(self.nb_test)
        ]  # used for seeding environment and agent

    def tearDown(self) -> None:
        self.envref.close()

    def test_auxilliary_func(self):
        """
        test the methods _compute_sub_pos works
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            MyClass = EducPandaPowerBackend.init_grid(self.envref.backend)
            backend = MyClass()

        backend.n_sub = self.envref.backend.n_sub
        backend.n_load = self.envref.backend.n_load
        backend.n_gen = self.envref.backend.n_gen
        backend.n_line = self.envref.backend.n_line
        backend.gen_to_subid = self.envref.backend.gen_to_subid
        backend.load_to_subid = self.envref.backend.load_to_subid
        backend.line_or_to_subid = self.envref.backend.line_or_to_subid
        backend.line_ex_to_subid = self.envref.backend.line_ex_to_subid

        # now "hack" the class to check that the correct element are correctly implemented
        bk_cls = type(backend)

        # delete the attributes we want to test
        bk_cls.sub_info = None
        bk_cls.load_to_sub_pos = None
        bk_cls.gen_to_sub_pos = None
        bk_cls.line_or_to_sub_pos = None
        bk_cls.line_ex_to_sub_pos = None
        bk_cls.line_ex_to_sub_pos = None
        bk_cls.load_pos_topo_vect = None
        bk_cls.gen_pos_topo_vect = None
        bk_cls.line_or_pos_topo_vect = None
        bk_cls.line_ex_pos_topo_vect = None

        # test that the grid is not correct now
        with self.assertRaises(EnvError):
            bk_cls.assert_grid_correct_cls()

        # fill the _compute_sub_elements
        bk_cls._compute_sub_elements()
        assert np.sum(bk_cls.sub_info) == 56
        assert np.all(bk_cls.sub_info == [3, 6, 4, 6, 5, 6, 3, 2, 5, 3, 3, 3, 4, 3])

        # fill the *sub_pos
        bk_cls._compute_sub_pos()
        assert np.all(bk_cls.load_to_sub_pos == 0)
        assert np.all(bk_cls.gen_to_sub_pos == [1, 1, 1, 0, 0])
        assert np.all(
            bk_cls.line_or_to_sub_pos
            == [1, 2, 2, 3, 4, 2, 1, 2, 3, 4, 1, 2, 1, 1, 1, 2, 3, 1, 0, 3]
        )
        assert np.all(
            bk_cls.line_ex_to_sub_pos
            == [5, 2, 3, 4, 3, 5, 4, 1, 2, 2, 2, 1, 2, 3, 2, 1, 4, 5, 1, 2]
        )

        # fill the *pos_topo_vect
        backend._compute_pos_big_topo()  # i test the object class here
        assert np.all(
            bk_cls.load_pos_topo_vect == [3, 9, 13, 19, 24, 35, 40, 43, 46, 49, 53]
        )
        assert np.all(bk_cls.gen_pos_topo_vect == [4, 10, 25, 33, 0])
        assert np.all(
            bk_cls.line_or_pos_topo_vect
            == [
                1,
                2,
                5,
                6,
                7,
                11,
                14,
                26,
                27,
                28,
                36,
                37,
                41,
                47,
                50,
                15,
                16,
                20,
                30,
                38,
            ]
        )
        assert np.all(
            bk_cls.line_ex_pos_topo_vect
            == [
                8,
                21,
                12,
                17,
                22,
                18,
                23,
                44,
                48,
                51,
                42,
                54,
                45,
                52,
                55,
                31,
                39,
                29,
                34,
                32,
            ]
        )
        # this should pass
        bk_cls.assert_grid_correct_cls()
    
    def test_topo_vect_element(self):
        """
        .. newinversion:: 1.10.0
            Test this utilitary function 
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make(
                "educ_case14_storage",
                test=True,
                _add_to_name=type(self).__name__+"test_gridobjects_testauxfunctions",
            )
        cls = type(env)
        for el_id, el_pos_topo_vect in enumerate(cls.load_pos_topo_vect):
            res = cls.topo_vect_element(el_pos_topo_vect)
            assert "load_id" in res
            assert res["load_id"] == el_id
        for el_id, el_pos_topo_vect in enumerate(cls.gen_pos_topo_vect):
            res = cls.topo_vect_element(el_pos_topo_vect)
            assert "gen_id" in res
            assert res["gen_id"] == el_id
        for el_id, el_pos_topo_vect in enumerate(cls.storage_pos_topo_vect):
            res = cls.topo_vect_element(el_pos_topo_vect)
            assert "storage_id" in res
            assert res["storage_id"] == el_id
        for el_id, el_pos_topo_vect in enumerate(cls.line_or_pos_topo_vect):
            res = cls.topo_vect_element(el_pos_topo_vect)
            assert "line_id" in res
            assert res["line_id"] == {"or": el_id}
            assert "line_or_id" in res
            assert res["line_or_id"] == el_id
        for el_id, el_pos_topo_vect in enumerate(cls.line_ex_pos_topo_vect):
            res = cls.topo_vect_element(el_pos_topo_vect)
            assert "line_id" in res
            assert res["line_id"] == {"ex": el_id}
            assert "line_ex_id" in res
            assert res["line_ex_id"] == el_id
            

if __name__ == "__main__":
    unittest.main()
