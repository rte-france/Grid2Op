import numpy as np
import itertools

try:
    from .Action import HelperAction
    from .Exceptions import Grid2OpException
except (ModuleNotFoundError, ImportError):
    from Action import HelperAction
    from Exceptions import Grid2OpException

import pdb


class Converter(HelperAction):
    def __init__(self, action_space):
        HelperAction.__init__(self, action_space, action_space.legal_action, action_space.subtype)
        self.space_prng = action_space.space_prng
        self.seed_used = action_space.seed_used

    def init_actions(self):
        pass

    def convert_obs(self, obs):
        return obs

    def convert_act(self, act):
        return act


class IdToAct(Converter):
    def __init__(self, action_space):
        Converter.__init__(self, action_space)
        self.all_actions = []
        self.all_actions.append(super().__call__())  # add the do nothing topology
        self.n = 1

    def init_actions(self, all_actions=None):
        if all_actions is None:
            self.all_actions = []
            if "_set_line_status" in self.template_act.attr_list_vect:
                # powerline switch: disconnection
                for i in range(self.n_line):
                    self.all_actions.append(self.disconnect_powerline(line_id=i))

                # powerline switch: reconnection
                for bus_or in [1, 2]:
                    for bus_ex in [1, 2]:
                        for i in range(self.n_line):
                            tmp_act = self.reconnect_powerline(line_id=i, bus_ex=bus_ex, bus_or=bus_or)
                            self.all_actions.append(tmp_act)

            if "_set_topo_vect" in self.template_act.attr_list_vect:
                # topologies using the 'set' method
                self.all_actions += self.get_all_unitary_topologies_set(self)
            elif "_change_bus_vect" in self.template_act.attr_list_vect:
                # topologies 'change'
                self.all_actions += self.get_all_unitary_topologies_change(self)

        else:
            self.all_actions = all_actions
        self.n = len(self.all_actions)

    def sample(self):
        idx = self.space_prng.randint(0, self.n)
        return idx

    def convert_act(self, act):
        """
        In this converter, we suppose that "act" is an id, that is the output of a

        Parameters
        ----------
        act: ``int``

        Returns
        -------

        """

        return self.all_actions[act]


class ToVect(Converter):
    def __init__(self, action_space):
        Converter.__init__(self, action_space)
        self.do_nothing_vect = action_space({}).to_vect()

    def convert_obs(self, obs):
        return obs.to_vect()

    def convert_act(self, act):
        res = self.__call__({})
        res.from_vect(act)
        return res