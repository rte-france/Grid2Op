# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import warnings

import pdb

from grid2op.Exceptions import AmbiguousAction
from grid2op.Action.BaseAction import BaseAction


class VoltageOnlyAction(BaseAction):
    """
    This class is here to serve as a base class for the controler of the voltages (if any). It allows to perform
    only modification of the generator voltage set point.

    Only action of type "injection" are supported, and only setting "prod_v" keyword.
    """

    def __init__(self, gridobj):
        """
        See the definition of :func:`BaseAction.__init__` and of :class:`BaseAction` for more information. Nothing more is done
        in this constructor.

        """
        BaseAction.__init__(self, gridobj)
        self.authorized_keys = {"injection"}
        self.attr_list_vect = ["prod_v"]
        if self.shunts_data_available:
            self.attr_list_vect += ["shunt_p", "shunt_q", "shunt_bus"]
            self.authorized_keys.add("shunt")

    def _check_dict(self):
        """
        Check that nothing, beside prod_v has been updated with this action.

        Returns
        -------

        """
        if self._dict_inj:
            for el in self._dict_inj:
                if el not in self.attr_list_vect:
                    raise AmbiguousAction("Impossible to modify something different than \"prod_v\" using "
                                          "\"VoltageOnlyAction\" action.")

    def update(self, dict_):
        """
        As its original implementation, this method allows modifying the way a dictionary can be mapped to a valid
        :class:`BaseAction`.

        It has only minor modifications compared to the original :func:`BaseAction.update` implementation, most notably, it
        doesn't update the :attr:`BaseAction._dict_inj`. It raises a warning if attempting to change them.

        Parameters
        ----------
        dict_: :class:`dict`
            See the help of :func:`BaseAction.update` for a detailed explanation. **NB** all the explanations concerning the
            "injection", "change bus", "set bus", or "change line status" are irrelevant for this subclass.

        Returns
        -------
        self: :class:`PowerLineSet`
            Return object itself thus allowing multiple calls to "update" to be chained.

        """
        self._reset_vect()

        if dict_ is not None:
            for kk in dict_.keys():
                if kk not in self.authorized_keys:
                    warn = "The key \"{}\" used to update an action will be ignored. Valid keys are {}"
                    warn = warn.format(kk, self.authorized_keys)
                    warnings.warn(warn)

            self._digest_injection(dict_)
            self._digest_shunt(dict_)
            self._check_dict()
        return self
