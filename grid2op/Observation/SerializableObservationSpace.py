# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from grid2op.Space import SerializableSpace
from grid2op.Observation.CompleteObservation import CompleteObservation


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
    def __init__(self, gridobj, observationClass=CompleteObservation):
        """

        Parameters
        ----------
        gridobj: :class:`grid2op.Space.GridObjects`
            Representation of the objects in the powergrid.

        observationClass: ``type``
            Type of action used to build :attr:`Space.SerializableSpace._template_obj`

        """
        SerializableSpace.__init__(self, gridobj=gridobj, subtype=observationClass)
        self.observationClass = self.subtype
        self._empty_obs = self._template_obj

    @staticmethod
    def from_dict(dict_):
        """
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
        res = SerializableObservationSpace(gridobj=tmp,
                                           observationClass=tmp.subtype)
        return res
