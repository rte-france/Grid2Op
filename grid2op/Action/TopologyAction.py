import warnings
import pdb

from grid2op.Exceptions import AmbiguousAction
from grid2op.Action.BaseAction import BaseAction


class TopologyAction(BaseAction):
    """
    This class is model only topological actions.
    It will throw an ":class:`grid2op.Exception.AmbiguousAction`" error it someone attempt to change injections
    in any ways.

    It has the same attributes as its base class :class:`BaseAction`.

    It is also here to show an example on how to implement a valid class deriving from :class:`BaseAction`.

    """

    def __init__(self, gridobj):
        """
        See the definition of :func:`BaseAction.__init__` and of :class:`BaseAction` for more information. Nothing more is done
        in this constructor.

        """
        BaseAction.__init__(self, gridobj)

        # the injection keys is not authorized, meaning it will send a warning is someone try to implement some
        # modification injection.
        self.authorized_keys = set([k for k in self.authorized_keys if k != "injection" and k != "redispatch" and
                                                                       k != "hazards" and k != "maintenance"])

        self.attr_list_vect = ["_set_line_status", "_switch_line_status",
                               "_set_topo_vect", "_change_bus_vect"]

    def __call__(self):
        """
        Compare to the ancestor :func:`BaseAction.__call__` this type of BaseAction doesn't allow to change the injections.
        The only difference is in the returned value *dict_injection* that is always an empty dictionary.

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
            Thie array is :attr:`BaseAction._redispatch`

        """
        if self._dict_inj:
            raise AmbiguousAction("You asked to modify the injection with an action of class \"TopologyAction\".")
        self._check_for_ambiguity()
        return {}, self._set_line_status, self._switch_line_status, self._set_topo_vect, self._change_bus_vect,\
               self._redispatch

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
            "injection" part is irrelevant for this subclass.

        Returns
        -------
        self: :class:`TopologyAction`
            Return object itself thus allowing multiple calls to "update" to be chained.

        """
        self._reset_vect()

        if dict_ is not None:
            for kk in dict_.keys():
                if kk not in self.authorized_keys:
                    warn = "The key \"{}\" used to update an action will be ignored. Valid keys are {}"
                    warn = warn.format(kk, self.authorized_keys)
                    warnings.warn(warn)

            # self._digest_injection(dict_)  # only difference compared to the base class implementation.
            self._digest_setbus(dict_)
            self._digest_change_bus(dict_)
            self._digest_set_status(dict_)
            self._digest_change_status(dict_)
        return self

    def sample(self, space_prng):
        """
        Sample a Topology action.

        This method is not implemented at the moment. TODO

        Parameters
        ----------
        space_prng

        Returns
        -------
        res: :class:`TopologyAction`
            The current action (useful to chain some calls to methods)
        """
        self.reset()
        return self
