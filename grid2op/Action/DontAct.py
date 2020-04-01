from grid2op.Action.Action import Action


class DontAct(Action):
    """
    This class is model the action where you force someone to do absolutely nothing.

    """

    def __init__(self, gridobj):
        """
        See the definition of :func:`Action.__init__` and of :class:`Action` for more information. Nothing more is done
        in this constructor.

        """
        Action.__init__(self, gridobj)

        # the injection keys is not authorized, meaning it will send a warning is someone try to implement some
        # modification injection.
        self.authorized_keys = set()
        self.attr_list_vect = []

    def update(self, dict_):
        return self

    def sample(self, space_prng):
        return self
