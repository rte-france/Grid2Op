# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import numpy as np
from grid2op.Exceptions import IllegalAction
from grid2op.Rules.BaseRules import BaseRules


class PreventDiscoStorageModif(BaseRules):
    """
    This subclass only check that the action do not modify the storage power (charge / discharge) of a disconnected
    storage unit.

    See :func:`BaseRules.__call__` for a definition of the parameters of this function.

    """

    def __call__(self, action, env):
        """
        See :func:`BaseRules.__call__` for a definition of the parameters of this function.
        
        ..versionchanged:: 1.10.2
            In grid2op 1.10.2 this function is not called when the environment is reset:
            The "action" made by the environment to set the environment in the desired state is always legal
            
        """
        env_cls = type(env)
        if env_cls.n_storage == 0:
            # nothing to do if no storage
            return True, None

        # at first iteration, env.current_obs is None...
        storage_disco = env.backend.get_topo_vect()[env_cls.storage_pos_topo_vect] < 0
        storage_power, storage_set_bus, storage_change_bus = action.get_storage_modif()

        power_modif_disco = (np.isfinite(storage_power[storage_disco]) & 
                             (np.abs(storage_power[storage_disco]) >= 1e-7))
        not_set_status = storage_set_bus[storage_disco] <= 0
        not_change_status = ~storage_change_bus[storage_disco]
        if (power_modif_disco & not_set_status & not_change_status).any():
            tmp_ = power_modif_disco & not_set_status & not_change_status
            return False, IllegalAction(
                f"Attempt to modify the power produced / absorbed by a storage unit "
                f"without reconnecting it (check storage with id {(tmp_).nonzero()[0]}."
            )
        return True, None
