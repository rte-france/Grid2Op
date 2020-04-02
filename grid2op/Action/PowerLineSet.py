import numpy as np
import warnings
import itertools

import pdb

from grid2op.Exceptions import AmbiguousAction
from grid2op.Action.BaseAction import BaseAction


class PowerLineSet(BaseAction):
    """
    This class is here to model only a subpart of Topological actions, the one consisting of topological switching.
    It will throw an "AmbiguousAction" error if someone attempts to change injections in any way.

    It has the same attributes as its base class :class:`BaseAction`.

    It is also here to show an example of how to implement a valid class deriving from :class:`BaseAction`.

    **NB** This class doesn't allow to connect an object to other buses than their original bus. In this case,
    reconnecting a powerline cannot be considered "ambiguous": all powerlines are reconnected on bus 1 on both
    of their substations.

    """

    def __init__(self, gridobj):
        """
        See the definition of :func:`BaseAction.__init__` and of :class:`BaseAction` for more information. Nothing more is done
        in this constructor.

        """
        BaseAction.__init__(self, gridobj)

        # the injection keys is not authorized, meaning it will send a warning is someone try to implement some
        # modification injection.
        self.authorized_keys = set([k for k in self.authorized_keys
                                    if k != "injection" and k != "set_bus" and
                                    k != "change_bus" and k != "change_line_status" and
                                    k != "redispatch"])

        self.attr_list_vect = ["_set_line_status"]

    def __call__(self):
        """
        Compare to the ancestor :func:`BaseAction.__call__` this type of BaseAction doesn't allow to change the injections.
        The only difference is in the returned value *dict_injection* that is always an empty dictionary.

        Returns
        -------
        dict_injection: :class:`dict`
            This dictionary is always empty

        set_line_status: :class:`numpy.array`, dtype:int
            This array is :attr:`BaseAction._set_line_status`

        switch_line_status: :class:`numpy.array`, dtype:bool
            This array is :attr:`BaseAction._switch_line_status`, it is never modified

        set_topo_vect: :class:`numpy.array`, dtype:int
            This array is :attr:`BaseAction._set_topo_vect`, it is never modified

        change_bus_vect: :class:`numpy.array`, dtype:bool
            This array is :attr:`BaseAction._change_bus_vect`, it is never modified

        redispatch: :class:`numpy.ndarray`, dtype:float
            Always 0 for this class

        shunts: ``dict``
            Always empty for this class
        """
        if self._dict_inj:
            raise AmbiguousAction("You asked to modify the injection with an action of class \"TopologyAction\".")
        self._check_for_ambiguity()
        return {}, self._set_line_status, self._switch_line_status, self._set_topo_vect, self._change_bus_vect, \
               self._redispatch, {}

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

            self._digest_set_status(dict_)
            self._digest_hazards(dict_)
            self._digest_maintenance(dict_)
        self.disambiguate_reconnection()

        return self

    def check_space_legit(self):
        self.disambiguate_reconnection()
        self._check_for_ambiguity()

    def disambiguate_reconnection(self):
        """
        As this class doesn't allow to perform any topology change, when a powerline is reconnected, it's necessarily
        on the first bus of the substation.

        So it's not ambiguous in this case. We have to implement this logic here, and that is what is done in this
        function.

        """
        sel_ = self._set_line_status == 1
        if np.any(sel_):
            self._set_topo_vect[self.line_ex_pos_topo_vect[sel_]] = 1
            self._set_topo_vect[self.line_or_pos_topo_vect[sel_]] = 1

    def sample(self, space_prng):
        """
        Sample a PowerlineSwitch BaseAction.

        By default, this sampling will act on one random powerline, and it will either
        disconnect it or reconnect it each with equal probability.

        Parameters
        ----------
        space_prng: ``numpy.random.RandomState``
            The pseudo random number generator of the BaseAction space used to sample actions.

        Returns
        -------
        res: :class:`PowerLineSwitch`
            The sampled action
            
        """
        self.reset()
        # TODO here use the prng state from the ActionSpace !!!!
        i = space_prng.randint(0, self.size())  # the powerline on which we can act
        val = 2*np.random.randint(0, 2) - 1  # the action: +1 reconnect it, -1 disconnect it
        self._set_line_status[i] = val
        if val == 1:
            self._set_topo_vect[self.line_ex_pos_topo_vect[i]] = 1
            self._set_topo_vect[self.line_or_pos_topo_vect[i]] = 1
        return self
