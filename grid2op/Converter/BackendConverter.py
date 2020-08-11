# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

"""
In the RL4Grid framework, the order of the objects are not specified. They are :class:`RL4Grid.Backend` dependant.

The utilities in this modules allow to convert an action from one backend to another.
They can also be used to convert observation.

TODO coming soon.
"""
import copy
from grid2op.Backend import Backend


class BackendConverter(Backend):
    """
    Convert two instance of backend to "align" them.

    This means that grid2op will behave exactly as is the "source backend" class is used everywhere, but
    the powerflow computation will be carried out by the "target backend".

    This means that from grid2op point of view, and from the agent point of view, line will be the order given
    by "source backend", load will be in the order of "source backend", topology will be given with the
    one from "source backend" etc. etc.

    Be careful, the BackendAction will also need to be transformed. Backend action is given with the order
    of the "source backend" and will have to be modified when transmitted to the "target backend".

    On the other end, no powerflow at all (except if some powerflows are performed at the initialization) will
    be computed using the source backend, only the target backend is relevant for the powerflow computations.
    
    Note that these backend need to access the grid description file from both "source backend" and "target backend" 
    class. The underlying grid must be the same.
    """
    def __init__(self,
                 source_backend_class,
                 target_backend_class,
                 target_backend_grid_path,
                 detailed_infos_for_cascading_failures=False):
        Backend.__init__(self, detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures)
        difcf = detailed_infos_for_cascading_failures
        self.source_backend = source_backend_class(detailed_infos_for_cascading_failures=difcf)
        self.target_backend = target_backend_class(detailed_infos_for_cascading_failures=difcf)
        self.target_backend_grid_path = target_backend_grid_path

        # TODO
        self._line_tg2sr = None  # if tmp is from the target backend, then tmp[self._line_tg2sr] is ordered according to the source backend
        self._line_tg2sr = None

    def load_grid(self, path=None, filename=None):
        self.source_backend.load_grid(path, filename)
        #TODO define the self.n_line, self.n_sub etc.
        self.target_backend.load_grid(path=self.target_backend_grid_path)

    def assert_grid_correct_after_powerflow(self):
        self.source_backend.assert_grid_correct_after_powerflow()
        # TODO set the __class__ here
        self.target_backend.assert_grid_correct_after_powerflow()

    def reset(self, grid_path, grid_filename=None):
        """
        Reload the power grid.
        For backwards compatibility this method calls `Backend.load_grid`.
        But it is encouraged to overload it in the subclasses.
        """
        self.target_backend.reset(grid_path, grid_filename=None)

    def close(self):
        self.source_backend.close()
        self.target_backend.close()

    def apply_action(self, action):
        # action is from the source backend
        action_target = self._transform_action(action)
        self.target_backend.apply_action(action_target)

    def runpf(self, is_dc=False):
        return self.target_backend.runnpf(id_dc=is_dc)

    def copy(self):
        res = self
        res.target_backend_grid_path = copy.deepcopy(self.target_backend_grid_path)
        res.source_backend = res.source_backend.copy()
        res.target_backend = res.target_backend.copy()

    def save_file(self, full_path):
        self.target_backend.save_file(full_path)
        self.source_backend.save_file(full_path)

    def get_line_status(self):
        tmp = self.source_backend.get_line_status()
        return tmp[self._line_tg2sr]

    def get_line_flow(self):
        tmp = self.source_backend.get_line_flow()
        return tmp[self._line_tg2sr]

    def set_thermal_limit(self, limits):
        super().set_thermal_limit(limits=limits)
        self.target_backend.set_thermal_limit(limits=limits)
        self.source_backend.set_thermal_limit(limits=limits)


    def _transform_action(self, source_action):
        # transform the source action into the target backend action
        # TODO
        return source_action

