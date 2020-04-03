from grid2op.Action.BaseAction import BaseAction


class DontAct(BaseAction):
    """
    This class is model the action where you force someone to do absolutely nothing.

    """

    def __init__(self, gridobj):
        """
        See the definition of :func:`BaseAction.__init__` and of :class:`BaseAction` for more information. Nothing more is done
        in this constructor.

        """
        BaseAction.__init__(self, gridobj)

        # the injection keys is not authorized, meaning it will send a warning is someone try to implement some
        # modification injection.
        self.authorized_keys = set()
        self.attr_list_vect = []

    def update(self, dict_):
        return self

    def sample(self, space_prng):
        return self
