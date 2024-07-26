# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import warnings
from typing import Optional, Dict, Literal

from grid2op.Exceptions import AmbiguousAction
from grid2op.Action.baseAction import BaseAction


class PlayableAction(BaseAction):
    """
    From this class inherit all actions that the player will be allowed to do. This includes for example
    :class:`TopologyAndDispatchAction` or :class:`TopologyAction`
    """

    authorized_keys = {
        "set_line_status",
        "change_line_status",
        "set_bus",
        "change_bus",
        "redispatch",
        "flexibility",
        "set_storage",
        "curtail",
        "raise_alarm",
        "raise_alert"
    }

    attr_list_vect = [
        "_set_line_status",
        "_switch_line_status",
        "_set_topo_vect",
        "_change_bus_vect",
        "_redispatch",
        "_flexibility",
        "_storage_power",
        "_curtail",
        "_raise_alarm",
        "_raise_alert"
    ]
    attr_list_set = set(attr_list_vect)
    shunt_added = True  # no shunt here

    def __init__(self, _names_chronics_to_backend: Optional[Dict[Literal["loads", "prods", "lines"], Dict[str, str]]]=None):
        super().__init__(_names_chronics_to_backend)

        self.authorized_keys_to_digest = {
            "set_line_status": self._digest_set_status,
            "change_line_status": self._digest_change_status,
            "set_bus": self._digest_setbus,
            "change_bus": self._digest_change_bus,
            "redispatch": self._digest_redispatching,
            "flexibility": self._digest_flexibility,
            "set_storage": self._digest_storage,
            "curtail": self._digest_curtailment,
            "raise_alarm": self._digest_alarm,
            "raise_alert": self._digest_alert,
        }

    def __call__(self):
        """
         .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Compare to the ancestor :func:`BaseAction.__call__` this type of BaseAction doesn't allow internal actions
        The returned tuple is same, but with empty dictionaries for internal actions

        Returns
        -------
        dict_injection: ``dict``
            This dictionary is always empty

        set_line_status: :class:`numpy.ndarray`, dtype:int
            This array is :attr:`BaseAction._set_line_status`

        switch_line_status: :class:`numpy.ndarray`, dtype:bool
            This array is :attr:`BaseAction._switch_line_status`

        set_topo_vect: :class:`numpy.ndarray`, dtype:int
            This array is :attr:`BaseAction._set_topo_vect`

        change_bus_vect: :class:`numpy.ndarray`, dtype:bool
            This array is :attr:`BaseAction._change_bus_vect`

        redispatch: :class:`numpy.ndarray`, dtype:float
            The array is :attr:`BaseAction._redispatch`

        flexibility: :class:`numpy.ndarray`, dtype:float
            The array is :attr:`BaseAction._flexibility`

        curtail: :class:`numpy.ndarray`, dtype:float
            The array is :attr:`BaseAction._curtail`

        shunts: ``dict``
            Always empty for this class
        """
        if self._dict_inj:
            raise AmbiguousAction("Injections actions are not playable.")

        self._check_for_ambiguity()
        return (
            {},
            self._set_line_status,
            self._switch_line_status,
            self._set_topo_vect,
            self._change_bus_vect,
            self._redispatch,
            self._flexibility,
            self._storage_power,
            {},
        )

    def update(self, dict_):
        """
         .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Similar to :class:`BaseAction`, except that the allowed entries are limited to the playable action set

        Parameters
        ----------
        dict_: :class:`dict`
            See the help of :func:`BaseAction.update` for a detailed explanation. 
            If an entry is not in the playable action set, this will raise

        Returns
        -------
        self: :class:`PlayableAction`
            Return object itself thus allowing multiple calls to "update" to be chained.

        """

        self._reset_vect()
        warn_msg = (
            'The key "{}" used to update an action will be ignored. Valid keys are {}'
        )

        if dict_ is None:
            return self

        for kk in dict_.keys():
            if kk not in self.authorized_keys:
                warn = warn_msg.format(kk, self.authorized_keys)
                warnings.warn(warn)
            else:
                self.authorized_keys_to_digest[kk](dict_)

        return self
