# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import warnings
from grid2op.Converter import BackendConverter
from grid2op.Backend import PandaPowerBackend

from grid2op.tests.helper_path_test import *
from grid2op.Agent import AgentWithConverter, MLAgent
from grid2op.Converter import IdToAct
from grid2op.Rules import AlwaysLegal
from grid2op import make
from grid2op.Parameters import Parameters


class TestLoading(HelperTests):
    def test_init(self):
        backend = BackendConverter(source_backend_class=PandaPowerBackend,
                                   target_backend_class=PandaPowerBackend,
                                   target_backend_grid_path=None)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = make(test=True, backend=backend)


if __name__ == "__main__":
    unittest.main()
