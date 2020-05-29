# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

"""
This file contains the settings (path to the case file, chronics converter etc. that allows to run
the competition "L2RPN 2019" that took place on the pypownet plateform.

It is present to reproduce this competition.
"""
import os
import warnings
from pathlib import Path
import numpy as np

from grid2op.Action import BaseAction
from grid2op.Exceptions import AmbiguousAction, IncorrectNumberOfElements
from grid2op.Chronics.ReadPypowNetData import ReadPypowNetData  # imported by another module

file_dir = Path(__file__).parent.absolute()
grid2op_root = file_dir.parent.absolute()
grid2op_root = str(grid2op_root)
dat_dir = os.path.abspath(os.path.join(grid2op_root, "data"))
case_dir = "l2rpn_2019"
grid_file = "grid.json"

L2RPN2019_CASEFILE = os.path.join(dat_dir, case_dir, grid_file)
L2RPN2019_CHRONICSPATH = os.path.join(dat_dir, case_dir, "chronics")


CASE_14_L2RPN2019_LAYOUT = graph_layout = [(-280, -81), (-100, -270), (366, -270), (366, -54), (-64, -54), (-64, 54),
                                           (450, 0), (550, 0), (326, 54), (222, 108), (79, 162), (-170, 270),
                                           (-64, 270), (222, 216)]

# names of object of the grid were not in the same order as the default one
L2RPN2019_DICT_NAMES = {'loads': {'2_C-10.61': 'load_1_0',
                                       '3_C151.15': 'load_2_1',
                                       '14_C63.6': 'load_13_10',
                                       '4_C-9.47': 'load_3_2',
                                       '5_C201.84': 'load_4_3',
                                       '6_C-6.27': 'load_5_4',
                                       '9_C130.49': 'load_8_5',
                                       '10_C228.66': 'load_9_6',
                                       '11_C-138.89': 'load_10_7',
                                       '12_C-27.88': 'load_11_8',
                                       '13_C-13.33': 'load_12_9'},
                             'lines': {'1_2_1': '0_1_0',
                                       '1_5_2': '0_4_1',
                                       '9_10_16': '8_9_16',
                                       '9_14_17': '8_13_15',
                                       '10_11_18': '9_10_17',
                                       '12_13_19': '11_12_18',
                                       '13_14_20': '12_13_19',
                                       '2_3_3': '1_2_2',
                                       '2_4_4': '1_3_3',
                                       '2_5_5': '1_4_4',
                                       '3_4_6': '2_3_5',
                                       '4_5_7': '3_4_6',
                                       '6_11_11': '5_10_12',
                                       '6_12_12': '5_11_11',
                                       '6_13_13': '5_12_10',
                                       '4_7_8': '3_6_7',
                                       '4_9_9': '3_8_8',
                                       '5_6_10': '4_5_9',
                                       '7_8_14': '6_7_13',
                                       '7_9_15': '6_8_14'},
                             'prods': {'1_G137.1': 'gen_0_4',
                                       '3_G36.31': 'gen_1_0',
                                       '6_G63.29': 'gen_2_1',
                                       '2_G-56.47': 'gen_5_2',
                                       '8_G40.43': 'gen_7_3'}}


# class of the action didn't implement the "set" part. Only change was present.
# Beside when reconnected, objects were always reconnected on bus 1.
# This is not used at the moment.
class L2RPN2019_Action(BaseAction):
    """
    This class is here to model only a subpart of Topological actions, the one consisting in topological switching.
    It will throw an "AmbiguousAction" error it someone attempt to change injections in any ways.

    It has the same attributes as its base class :class:`BaseAction`.

    It is also here to show an example on how to implement a valid class deriving from :class:`BaseAction`.

    **NB** This class doesn't allow to connect object to other buses than their original bus. In this case,
    reconnecting a powerline cannot be considered "ambiguous". We have to
    """
    def __init__(self):
        """
        See the definition of :func:`BaseAction.__init__` and of :class:`BaseAction` for more information. Nothing more is done
        in this constructor.
        """
        BaseAction.__init__(self)

        # the injection keys is not authorized, meaning it will send a warning is someone try to implement some
        # modification injection.
        self.authorized_keys = set([k for k in self.authorized_keys
                                    if k != "injection" and k != "set_bus" and "set_line_status"])

    def __call__(self):
        """
        Compare to the ancestor :func:`BaseAction.__call__` this type of BaseAction doesn't allow to change the injections.
        The only difference is in the returned value *dict_injection* that is always an empty dictionnary.

        Returns
        -------
        dict_injection: :class:`dict`
            This dictionnary is always empty

        set_line_status: :class:`numpy.array`, dtype:int
            This array is :attr:`BaseAction._set_line_status`

        switch_line_status: :class:`numpy.array`, dtype:bool
            This array is :attr:`BaseAction._switch_line_status`, it is never modified

        set_topo_vect: :class:`numpy.array`, dtype:int
            This array is :attr:`BaseAction._set_topo_vect`, it is never modified

        change_bus_vect: :class:`numpy.array`, dtype:bool
            This array is :attr:`BaseAction._change_bus_vect`, it is never modified
        """
        if self._dict_inj:
            raise AmbiguousAction("You asked to modify the injection with an action of class \"TopologyAction\".")
        self._check_for_ambiguity()
        return {}, self._set_line_status, self._switch_line_status, self._set_topo_vect, self._change_bus_vect

    def update(self, dict_):
        """
        As its original implementation, this method allows to modify the way a dictionnary can be mapped to a valid
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
            Return object itself thus allowing mutiple call to "update" to be chained.
        """

        self.as_vect = None
        if dict_ is not None:
            for kk in dict_.keys():
                if kk not in self.authorized_keys:
                    warn = "The key \"{}\" used to update an action will be ignored. Valid keys are {}"
                    warn = warn.format(kk, self.authorized_keys)
                    warnings.warn(warn)

            self._digest_change_bus(dict_)
            self._digest_hazards(dict_)
            self._digest_maintenance(dict_)
            self._digest_change_status(dict_)

        # self.disambiguate_reconnection()

        return self

    def size(self):
        """
        Compare to the base class, this action has a shorter size, as all information about injections are ignored.
        Returns
        -------
        size: ``int``
            The size of :class:`PowerLineSet` converted to an array.
        """
        return self.n_line + self.dim_topo

    def to_vect(self):
        """
        See :func:`BaseAction.to_vect` for a detailed description of this method.

        This method has the same behaviour as its base class, except it doesn't require any information about the
        injections to be sent, thus being more efficient from a memory footprint perspective.

        Returns
        -------
        _vectorized: :class:`numpy.array`, dtype:float
            The instance of this action converted to a vector.
        """
        if self.as_vect is None:
            self.as_vect = np.concatenate((
                self._switch_line_status.flatten().astype(np.float),
                self._change_bus_vect.flatten().astype(np.float)
                                           ))

            if self.as_vect.shape[0] != self.size():
                raise AmbiguousAction("L2RPN2019_Action has not the proper shape.")

        return self.as_vect

    def from_vect(self, vect):
        """
        See :func:`BaseAction.from_vect` for a detailed description of this method.

        Nothing more is made except the initial vector is (much) smaller.

        Parameters
        ----------
        vect: :class:`numpy.array`, dtype:float
            A vector reprenseting an instance of :class:`.`

        Returns
        -------

        """
        self.reset()
        if vect.shape[0] != self.size():
            raise IncorrectNumberOfElements("Incorrect number of elements found while loading a \"TopologyAction\" from a vector. Found {} elements instead of {}".format(vect.shape[1], self.size()))
        prev_ = 0
        next_ = self.n_line

        self._switch_line_status = vect[prev_:next_]
        self._switch_line_status = self._switch_line_status.astype(np.bool); prev_=next_; next_+= self.dim_topo
        self._change_bus_vect = vect[prev_:next_]
        self._change_bus_vect = self._change_bus_vect.astype(np.bool)

        # self.disambiguate_reconnection()

        self._check_for_ambiguity()

    def sample(self, space_prng):
        """
        Sample a PowerlineSwitch BaseAction.

        By default, this sampling will act on one random powerline, and it will either
        disconnect it or reconnect it each with equal probability.

        Parameters
        ----------
        space_prng

        Returns
        -------
        res: :class:`PowerLineSwitch`
            The sampled action
        """
        self.reset()
        i = np.random.randint(0, self.size())  # the powerline on which we can act
        val = 2*np.random.randint(0, 2) - 1  # the action: +1 reconnect it, -1 disconnect it
        self._set_line_status[i] = val
        if val == 1:
            self._set_topo_vect[self.line_ex_pos_topo_vect[i]] = 1
            self._set_topo_vect[self.line_or_pos_topo_vect[i]] = 1
        return self
