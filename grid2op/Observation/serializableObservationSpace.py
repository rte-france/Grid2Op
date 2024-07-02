# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import logging
import copy

from grid2op.Space import SerializableSpace
from grid2op.Observation.completeObservation import CompleteObservation


class SerializableObservationSpace(SerializableSpace):
    """
    This class allows to serialize / de serialize the action space.

    It should not be used inside an Environment, as some functions of the action might not be compatible with
    the serialization, especially the checking of whether or not an BaseObservation is legal or not.

    Attributes
    ----------
    observationClass: ``type``
        Type used to build the :attr:`SerializableActionSpace._template_act`

    _empty_obs: :class:`BaseObservation`
        An instance of the "*observationClass*" provided used to provide higher level utilities

    """

    def __init__(self, gridobj, observationClass=CompleteObservation, logger=None, _init_grid=True, _local_dir_cls=None):
        """

        Parameters
        ----------
        gridobj: :class:`grid2op.Space.GridObjects`
            Representation of the objects in the powergrid.

        observationClass: ``type``
            Type of action used to build :attr:`Space.SerializableSpace._template_obj`

        """
        SerializableSpace.__init__(
            self, gridobj=gridobj,
            subtype=observationClass,
            _init_grid=_init_grid,
            _local_dir_cls=_local_dir_cls
        )
        self.observationClass = self.subtype
        self._empty_obs = self._template_obj
        
        if logger is None:
            self.logger = logging.getLogger(__name__)
            self.logger.disabled = True
        else:
            self.logger: logging.Logger = logger.getChild("grid2op_ObsSpace")

    def _custom_deepcopy_for_copy(self, new_obj):
        super()._custom_deepcopy_for_copy(new_obj)
        # SerializableObservationSpace
        new_obj.observationClass = self.observationClass  # const
        new_obj._empty_obs = self._template_obj  # const
        new_obj.logger = copy.deepcopy(self.logger)

    @staticmethod
    def from_dict(dict_):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\
            This is used internally by EpisodeData to restore the state of the powergrid

        Allows the de-serialization of an object stored as a dictionary (for example in the case of json saving).

        Parameters
        ----------
        dict_: ``dict``
            Representation of an BaseObservation Space (aka SerializableObservationSpace) as a dictionary.

        Returns
        -------
        res: :class:``SerializableObservationSpace``
            An instance of an action space matching the dictionary.

        """
        tmp = SerializableSpace.from_dict(dict_)
        CLS = SerializableObservationSpace.init_grid(tmp)
        res = CLS(gridobj=tmp, observationClass=tmp.subtype, _init_grid=False)
        return res

    def get_indx_extract(self, attr_name):
        # backward compatibility (due to consistency with previous names)
        if attr_name == "prod_p":
            attr_name = "gen_p"
        elif attr_name == "prod_q":
            attr_name = "gen_q"
        elif attr_name == "prod_v":
            attr_name = "gen_v"
        return super().get_indx_extract(attr_name)
