# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import warnings

from grid2op.Exceptions import *
from grid2op.Action.BaseAction import BaseAction
from grid2op.Action.SerializableActionSpace import SerializableActionSpace


class ActionSpace(SerializableActionSpace):
    """
    :class:`ActionSpace` should be created by an :class:`grid2op.Environment.Environment`
    with its parameters coming from a properly
    set up :class:`grid2op.Backend.Backend` (ie a Backend instance with a loaded powergrid.
    See :func:`grid2op.Backend.Backend.load_grid` for
    more information).

    It will allow, thanks to its :func:`ActionSpace.__call__` method to create valid :class:`BaseAction`. It is the
    the preferred way to create an object of class :class:`BaseAction` in this package.

    On the contrary to the :class:`BaseAction`, it is NOT recommended to overload this helper. If more flexibility is
    needed on the type of :class:`BaseAction` created, it is recommended to pass a different "*actionClass*" argument
    when it's built. Note that it's mandatory that the class used in the "*actionClass*" argument derived from the
    :class:`BaseAction`.

    Attributes
    ----------
    legal_action: :class:`grid2op.RulesChecker.BaseRules`
        Class specifying the rules of the game used to check the legality of the actions.

    """
    
    def __init__(self, gridobj, legal_action, actionClass=BaseAction):
        """
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

            The actions space is created by the environment. Do not attempt to create one yourself.

        All parameters (name_gen, name_load, name_line, sub_info, etc.) are used to fill the attributes having the
        same name. See :class:`ActionSpace` for more information.

        Parameters
        ----------

        gridobj: :class:`grid2op.Space.GridObjects`
            The representation of the powergrid.

        actionClass: ``type``
            Note that this parameter expected a class and not an object of the class. It is used to return the
            appropriate action type.

        legal_action: :class:`grid2op.RulesChecker.BaseRules`
            Class specifying the rules of the game used to check the legality of the actions.

        """
        SerializableActionSpace.__init__(self, gridobj, actionClass=actionClass)
        self.legal_action = legal_action

    def __call__(self, dict_=None, check_legal=False, env=None):
        """
        This utility allows you to build a valid action, with the proper sizes if you provide it with a valid
        dictionary.

        More information about this dictionary can be found in the :func:`Action.update` help. This dictionary
        is not changed in this method.

        **NB** This is the only recommended way to make a valid, with proper dimension :class:`Action` object:

        Examples
        --------
        Here is a short example on how to make a action. For more detailed examples see :func:`Action.update`

        .. code-block:: python

            import grid2op
            # create a simple environment
            env = grid2op.make()
            act = env.action_space({})
            # act is now the "do nothing" action, that doesn't modify the grid.

        Parameters
        ----------
        dict_ : ``dict``
            see :func:`Action.__call__` documentation for an extensive help about this parameter

        check_legal: ``bool``
            is there a test performed on the legality of the action. **NB** When an object of class :class:`Action` is
            used, it is automatically tested for ambiguity. If this parameter is set to ``True`` then a legality test
            is performed. An action can be illegal if the environment doesn't allow it, for example if an agent tries
            to reconnect a powerline during a maintenance.

        env: :class:`grid2op.Environment.Environment`, optional
            An environment used to perform a legality check.

        Returns
        -------
        res: :class:`BaseAction`
            An action that is valid and corresponds to what the agent want to do with the formalism defined in
            see :func:`Action.udpate`.

        """

        res = self.actionClass()
        # update the action
        res.update(dict_)
        if check_legal:
            is_legal, reason = self._is_legal(res, env)
            if not is_legal:
                raise reason

        return res

    def _is_legal(self, action, env):
        """
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

            Whether an action is legal or not is checked by the environment at each call
            to `env.step`

        Parameters
        ----------
        action: :class:`BaseAction`
            The action to test

        env: :class:`grid2op.Environment.Environment`
            The current environment

        Returns
        -------
        res: ``bool``
            ``True`` if the action is legal, ie is allowed to be performed by the rules of the game. ``False``
            otherwise.
        """
        if env is None:
            warnings.warn("Cannot performed legality check because no environment is provided.")
            return True
        is_legal, reason = self.legal_action(action, env)
        return is_legal, reason
