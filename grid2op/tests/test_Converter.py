# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import warnings
from grid2op.tests.helper_path_test import *

from grid2op.MakeEnv import make
from grid2op.Parameters import Parameters
from grid2op.Converter import ConnectivityConverter, IdToAct
import tempfile
import pdb

import warnings
warnings.simplefilter("error")


class TestConverter(HelperTests):
    def setUp(self):
        """
        The case file is a representation of the case14 as found in the ieee14 powergrid.
        :return:
        """
        param = Parameters()
        param.init_from_dict({"NO_OVERFLOW_DISCONNECTION": True})
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = make("rte_case14_redisp", test=True, param=param)
        np.random.seed(0)

    def tearDown(self):
        self.env.close()

    def test_ConnectivityConverter(self):
        converter = ConnectivityConverter(self.env.action_space)
        converter.init_converter()
        converter.seed(0)
        assert np.all(converter.subs_ids == np.array([ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  2,  2,
                                                       2,  2,  2,  2,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
                                                       3,  3,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  5,  5,  5,  5,  5,
                                                       5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  8,  8,  8,  8,  8,  8,  8,
                                                       8,  8,  8, 12, 12, 12, 12, 12, 12]))
        assert len(converter.obj_type) == converter.n
        assert len(set(converter.obj_type)) == converter.n
        assert converter.pos_topo.shape[0] == converter.n
        assert len(set([tuple(sorted(el)) for el in converter.pos_topo])) == converter.n

        coded_act = np.random.rand(converter.n)
        pred = converter._compute_disagreement(coded_act, np.ones(converter.n))
        assert np.abs( (converter.n - coded_act.sum())/converter.n - pred) <= self.tol_one
        pred = converter._compute_disagreement(coded_act, np.arange(converter.n))
        assert np.abs(coded_act.sum()/converter.n - pred) <= self.tol_one

        # and not test i can produce an action that can be implemented
        act = converter.convert_act(encoded_act=coded_act)
        obs, reward, done, info = self.env.step(act)

        # test sample
        obs = self.env.reset()
        act = converter.sample()
        obs, reward, done, info = self.env.step(act)

    def test_max_sub_changed(self):
        for ms_sub in [1, 2, 3]:
            converter = ConnectivityConverter(self.env.action_space)
            converter.init_converter(max_sub_changed=ms_sub)
            converter.seed(0)

            coded_act = np.random.rand(converter.n)

            # and not test i can produce an action that can be implemented
            act = converter.convert_act(encoded_act=coded_act)
            lines_impacted, subs_impacted = act.get_topological_impact()
            assert np.sum(subs_impacted) == ms_sub, "wrong number of substations affected. It should be {}".format(ms_sub)
            obs, reward, done, info = self.env.step(act)

            # test sample
            obs = self.env.reset()
            act = converter.sample()
            lines_impacted, subs_impacted = act.get_topological_impact()
            assert np.sum(subs_impacted) == ms_sub, "wrong number of substations affected. It should be {}".format(ms_sub)
            obs, reward, done, info = self.env.step(act)


class TestIdToAct(HelperTests):
    def setUp(self):
        """
        The case file is a representation of the case14 as found in the ieee14 powergrid.
        :return:
        """
        param = Parameters()
        param.init_from_dict({"NO_OVERFLOW_DISCONNECTION": True})
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = make("rte_case14_redisp", test=True, param=param)
        np.random.seed(0)

    def tearDown(self):
        self.env.close()

    def test_save_reload(self):
        path_ = tempfile.mkdtemp()
        converter = IdToAct(self.env.action_space)
        converter.init_converter(set_line_status=False, change_bus_vect=False)
        converter.save(path_, "tmp_convert.npy")
        init_size = converter.size()
        array = np.load(os.path.join(path_, "tmp_convert.npy"))
        act = converter.convert_act(27)
        act_ = converter.convert_act(-1)
        assert array.shape[1] == self.env.action_space.size()
        converter2 = IdToAct(self.env.action_space)
        converter2.init_converter(all_actions=os.path.join(path_, "tmp_convert.npy"))
        assert init_size == converter2.size()
        act2 = converter2.convert_act(27)
        act2_ = converter2.convert_act(-1)
        assert act == act2
        assert act_ == act2_


if __name__ == "__main__":
    unittest.main()
