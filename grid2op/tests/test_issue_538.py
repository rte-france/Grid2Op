# Copyright (c) 2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import tempfile
import grid2op
from grid2op.Runner import Runner
from grid2op.Episode import EpisodeData
import warnings
import unittest


class Issue538Tester(unittest.TestCase):
    def test_is_done(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("rte_case14_realistic", test=True, _add_to_name=type(self).__name__)
        obs = env.reset()
        with tempfile.TemporaryDirectory() as tmp_path:
            runner = Runner(**env.get_params_for_runner())
            li_ep_data_ref = runner.run(nb_episode=1, path_save=tmp_path, add_detailed_output=True, max_iter=10)
            
            li_ep = EpisodeData.list_episode(path_agent=tmp_path)
            ep_data_loaded = EpisodeData.from_disk(*li_ep[0])

        assert not li_ep_data_ref[0][-1].observations[0]._is_done
        assert not ep_data_loaded.observations[0]._is_done
        g1 = li_ep_data_ref[0][-1].observations[0].get_energy_graph()
        assert len(g1.nodes) == 14
        g2 = ep_data_loaded.observations[0].get_energy_graph()
        assert len(g2.nodes) == 14


if __name__ == "__main__":
    unittest.main()
