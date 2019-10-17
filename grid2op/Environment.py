"""
This module defines the :class:`Environment` the higher level representation of the world with which an
:class:`grid2op.Agent` will interact.

The environment receive an :class:`grid2op.Action` from the :class:`grid2op.Agent` in the :func:`Environment.step`
and returns an
:class:`grid2op.Observation` that the :class:`grid2op.Agent` will use to perform the next action.

An environment is better used inside a :class:`grid2op.Runner`, mainly because runners abstract the interaction
between environment and agent, and ensure the environment are properly reset after each epoch.
"""

import numpy as np
import time
import os

try:
    from .Action import HelperAction, Action, TopologyAction
    from .Exceptions import *
    from .Observation import CompleteObservation, ObservationHelper
    from .Reward import FlatReward, RewardHelper
    from .GameRules import GameRules, AllwaysLegal
except (ModuleNotFoundError, ImportError):
    from Action import HelperAction, Action, TopologyAction
    from Exceptions import *
    from Observation import CompleteObservation, ObservationHelper
    from Reward import ConstantReward, RewardHelper
    from GameRules import GameRules, AllwaysLegal

import pdb


# TODO code "start from a given time step"
class Environment:
    def __init__(self,
                 init_grid_path: str,
                 chronics_handler,
                 backend,
                 parameters,
                 names_chronics_to_backend=None,
                 actionClass=TopologyAction,
                 observationClass=CompleteObservation,
                 rewardClass=FlatReward,
                 legalActClass=AllwaysLegal):
        """

        Parameters
        ----------
        init_grid_path: ``str``
            Path where the grid is stored. Used to initialize the :class:`grid2op.Backend`

        chronics_handler
        backend
        parameters
        names_chronics_to_backend
        actionClass
        observationClass
        rewardClass
        legalActClass
        """
        # TODO documentation!!

        # some timers
        self._time_apply_act = 0
        self._time_powerflow = 0
        self._time_extract_obs = 0

        # define logger
        self.logger = None

        # and calendar data
        self.time_stamp = None
        self.nb_time_step = 0

        # specific to power system
        self.parameters = parameters

        self.rewardClass = rewardClass

        # backend
        self.init_grid_path = os.path.abspath(init_grid_path)
        self.backend = backend
        self.backend.load_grid(self.init_grid_path)  # the real powergrid of the environment
        self.backend.assert_grid_correct()

        # rules of the game
        self.game_rules = GameRules(legalActClass=legalActClass)

        # action helper
        # action affecting the _grid that will be made by the agent
        self.helper_action_player = HelperAction(name_prod=self.backend.name_prods,
                                       name_load=self.backend.name_loads,
                                       name_line=self.backend.name_lines,
                                          subs_info=self.backend.subs_elements,
                                          load_to_subid=self.backend.load_to_subid,
                                          gen_to_subid=self.backend.gen_to_subid,
                                          lines_or_to_subid=self.backend.lines_or_to_subid,
                                          lines_ex_to_subid=self.backend.lines_ex_to_subid, #####
                                          load_to_sub_pos=self.backend.load_to_sub_pos,
                                          gen_to_sub_pos=self.backend.gen_to_sub_pos,
                                          lines_or_to_sub_pos=self.backend.lines_or_to_sub_pos,
                                          lines_ex_to_sub_pos=self.backend.lines_ex_to_sub_pos, #####
                                          load_pos_topo_vect=self.backend.load_pos_topo_vect,
                                          gen_pos_topo_vect=self.backend.gen_pos_topo_vect,
                                          lines_or_pos_topo_vect=self.backend.lines_or_pos_topo_vect,
                                          lines_ex_pos_topo_vect=self.backend.lines_ex_pos_topo_vect,
                                          actionClass=actionClass,
                                                 game_rules=self.game_rules)

        # action that affect the _grid made by the environment.
        self.helper_action_env = HelperAction(name_prod=self.backend.name_prods,
                                       name_load=self.backend.name_loads,
                                       name_line=self.backend.name_lines,
                                          subs_info=self.backend.subs_elements,
                                          load_to_subid=self.backend.load_to_subid,
                                          gen_to_subid=self.backend.gen_to_subid,
                                          lines_or_to_subid=self.backend.lines_or_to_subid,
                                          lines_ex_to_subid=self.backend.lines_ex_to_subid, #####
                                          load_to_sub_pos=self.backend.load_to_sub_pos,
                                          gen_to_sub_pos=self.backend.gen_to_sub_pos,
                                          lines_or_to_sub_pos=self.backend.lines_or_to_sub_pos,
                                          lines_ex_to_sub_pos=self.backend.lines_ex_to_sub_pos, #####
                                          load_pos_topo_vect=self.backend.load_pos_topo_vect,
                                          gen_pos_topo_vect=self.backend.gen_pos_topo_vect,
                                          lines_or_pos_topo_vect=self.backend.lines_or_pos_topo_vect,
                                          lines_ex_pos_topo_vect=self.backend.lines_ex_pos_topo_vect,
                                          actionClass=Action,
                                                 game_rules=self.game_rules)

        self.helper_observation = ObservationHelper(n_gen=self.backend.n_generators,
                                                    n_load=self.backend.n_loads,
                                                    n_lines=self.backend.n_lines,
                                                    subs_info=self.backend.subs_elements,
                                                    load_to_subid=self.backend.load_to_subid,
                                                    gen_to_subid=self.backend.gen_to_subid,
                                                    lines_or_to_subid=self.backend.lines_or_to_subid,
                                                    lines_ex_to_subid=self.backend.lines_ex_to_subid, #####
                                                    load_to_sub_pos=self.backend.load_to_sub_pos,
                                                    gen_to_sub_pos=self.backend.gen_to_sub_pos,
                                                    lines_or_to_sub_pos=self.backend.lines_or_to_sub_pos,
                                                    lines_ex_to_sub_pos=self.backend.lines_ex_to_sub_pos, #####
                                                    load_pos_topo_vect=self.backend.load_pos_topo_vect,
                                                    gen_pos_topo_vect=self.backend.gen_pos_topo_vect,
                                                    lines_or_pos_topo_vect=self.backend.lines_or_pos_topo_vect,
                                                    lines_ex_pos_topo_vect=self.backend.lines_ex_pos_topo_vect,
                                                    observationClass=observationClass,
                                                    rewardClass=rewardClass,
                                                    env=self)
        self.current_obs = None

        # type of power flow to play
        self.no_overflow_disconnection = self.parameters.NO_OVERFLOW_DISCONNECTION  # if True, then it will not disconnect lines above their thermal limits
        self.timestep_overflow = np.zeros(shape=(self.backend.n_lines,), dtype=np.int)
        self.nb_timestep_overflow_allowed = np.full(shape=(self.backend.n_lines,),
                                                    fill_value=self.parameters.NB_TIMESTEP_POWERFLOW_ALLOWED)
        self.hard_overflow_threshold = self.parameters.HARD_OVERFLOW_THRESHOLD
        self.time_remaining_before_reconnection = np.full(shape=(self.backend.n_lines,), fill_value=0, dtype=np.int)
        self.env_dc = self.parameters.ENV_DC

        # handles input data
        self.chronics_handler = chronics_handler
        self.chronics_handler.initialize(self.backend.name_loads, self.backend.name_prods,
                                         self.backend.name_lines, self.backend.name_subs,
                                         names_chronics_to_backend=names_chronics_to_backend)
        self.names_chronics_to_backend = names_chronics_to_backend

        # test to make sure the backend is consistent with the chronics generator
        self.chronics_handler.check_validity(self.backend)

        # store environment modifications
        self.injection = None
        self.maintenance = None
        self.hazards = None

        # observation
        self.current_obs = None

        # reward
        self.reward_helper = RewardHelper(rewardClass=rewardClass)
        self.reward_helper.initialize(self)

        # performs one step to load the environment properly (first action need to be taken at first time step after
        # first injections given)
        do_nothing = self.helper_action_env({})
        *_, fail_to_start, _ = self.step(do_nothing)
        if fail_to_start:
            raise Grid2OpException("Impossible to initialize the powergrid, the powerflow diverge at iteration 0.")
        # test the backend returns object of the proper size
        self.backend.assert_grid_correct_after_powerflow()

        # for gym compatibility
        self.action_space = self.helper_action_player  # this should be an action !!!
        self.observation_space = self.helper_observation  # this return an observation.
        self.reward_range = self.reward_helper.range()
        self.viewer = None

    def reset_grid(self):
        """
        Reset the backend to a clean state by reloading the powergrid from the hard drive. This might takes some time.

        """
        self.backend.load_grid(self.init_grid_path)  # the real powergrid of the environment
        self.backend.assert_grid_correct()

        do_nothing = self.helper_action_env({})
        self.step(do_nothing)
        # test the backend returns object of the proper size
        self.backend.assert_grid_correct_after_powerflow()

    def add_text_logger(self, logger=None):
        """
        Add a text logger to this  :class:`Environment`

        Logging is for now an incomplete feature. It will get improved
        Parameters
        ----------
            seed:
               The logger to use

        """
        self.logger = logger
        return self

    def seed(self, seed=None):
        """
        Set the seed of this :class:`Environment` for a better control and to ease reproducible experiments.


        Parameters
        ----------
            seed: ``int``
               The seed to set.

        """
        # example from gym
        # self.np_random, seed = seeding.np_random(seed)
        # TODO make that more clean, see example of seeding @ https://github.com/openai/gym/tree/master/gym/utils
        self.chronics_handler.seed(seed)
        self.helper_observation.seed(seed)
        return [seed]

    def _update_actions(self):
        """
        Retrieve the actions to perform the update of the underlying powergrid represented by
         the :class:`grid2op.Backend`in the next time step.
        A call to this function will also read the next state of :attr:`chronics_handler`, so it must be called only
        once per time step.

        Returns
        --------
        res: :class:`grid2op.Action`
            The action representing the modification of the powergrid induced by the Backend.
        """
        timestamp, tmp = self.chronics_handler.next_time_step()
        if "injection" in tmp:
            self.injection = tmp["injection"]
        else:
            self.injection = None
        if 'maintenance' in tmp:
            self.maintenance = tmp['maintenance']
        else:
            self.maintenance = None
        if "hazards" in tmp:
            self.hazards = tmp["hazards"]
        else:
            self.hazards = None
        self.time_stamp = timestamp
        return self.helper_action_env({"injection": self.injection, "maintenance": self.maintenance, "hazards": self.hazards})

    def get_obs(self):
        """
        Return the observations of the current environment made by the :class:`grid2op.Agent`.

        Returns
        -------
        res: :class:`grid2op.Observation`
            The current Observation given to the :class:`grid2op.Agent` / bot / controler.
        """
        res = self.helper_observation(env=self)
        return res

    def _get_reward(self, action, has_error, is_done):
        return self.reward_helper(action, self, has_error, is_done)

    def _is_done(self, has_error, is_done):
        no_more_data = self.chronics_handler.done()
        return has_error or is_done or no_more_data

    def step(self, action):
        """
        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).

        Parameters
        ----------
            action: :class:`grid2op.Action`
                an action provided by the agent that is applied on the underlying through the backend.

        Returns
        -------
            observation: :class:`grid2op.Observation`
                agent's observation of the current environment
            reward: ``float``
                amount of reward returned after previous action
            done: ``bool``
                whether the episode has ended, in which case further step() calls will return undefined results
            info: ``dict``
                contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        has_error = True
        is_done = False
        disc_lines = None
        try:
            beg_ = time.time()
            self.backend.apply_action(action)
            env_modification = self._update_actions()
            self.backend.apply_action(env_modification)
            self._time_apply_act += time.time() - beg_

            self.nb_time_step += 1
            try:
                # compute the next _grid state
                beg_ = time.time()
                disc_lines, infos = self.backend.next_grid_state(env=self, is_dc=self.env_dc)
                self._time_powerflow += time.time() - beg_

                beg_ = time.time()
                overflow_lines = self.backend.get_line_overflow(self.current_obs)

                # one timestep passed, i can maybe reconnect some lines
                self.time_remaining_before_reconnection[self.time_remaining_before_reconnection > 0] -= 1

                # update the vector for lines that have been disconnected
                self.time_remaining_before_reconnection[disc_lines] = int(self.parameters.NB_TIMESTEP_RECONNECTION)

                # for the powerline that are on overflow, increase this time step
                self.timestep_overflow[overflow_lines] += 1

                # set to 0 the number of timestep for lines that are not on overflow
                self.timestep_overflow[~overflow_lines] = 0

                # build the observation
                self.current_obs = self.get_obs()
                self._time_extract_obs += time.time() - beg_

                has_error = False
            except Grid2OpException as e:
                if self.logger is not None:
                    self.logger.error("Impossible to compute next _grid state with error \"{}\"".format(e))
        except StopIteration:
            # episode is over
            is_done = True

        return self.current_obs, self._get_reward(action, has_error, is_done), self._is_done(has_error, is_done),\
               {"disc_lines": disc_lines}

    def reset(self):
        """
        Reset the environment to a clean state.
        It will reload the next chronics if any. And reset the grid to a clean state.

        This triggers a full reloading of both the chronics (if they are stored as files) and of the powergrid,
        to ensure the episode is fully over.

        This method should be called only at the end of an episode.
        """
        self.chronics_handler.next_chronics()
        self.chronics_handler.initialize(self.backend.name_loads, self.backend.name_prods,
                                         self.backend.name_lines, self.backend.name_subs,
                                         names_chronics_to_backend=self.names_chronics_to_backend)
        self.current_obs = None
        self.reset_grid()

        # if True, then it will not disconnect lines above their thermal limits
        self.no_overflow_disconnection = self.parameters.NO_OVERFLOW_DISCONNECTION
        self.timestep_overflow = np.zeros(shape=(self.backend.n_lines,), dtype=np.int)
        self.nb_timestep_overflow_allowed = np.full(shape=(self.backend.n_lines,),
                                                    fill_value=self.parameters.NB_TIMESTEP_POWERFLOW_ALLOWED)
        self.nb_time_step = 0
        self.hard_overflow_threshold = self.parameters.HARD_OVERFLOW_THRESHOLD
        self.time_remaining_before_reconnection = np.full(shape=(self.backend.n_lines,), fill_value=0, dtype=np.int)
        self.env_dc = self.parameters.ENV_DC

        self._time_apply_act = 0
        self._time_powerflow = 0
        self._time_extract_obs = 0
        return self.get_obs()

    def render(self, mode='human'):
        # TODO here, and reuse pypownet
        pass

    def close(self):
        # todo there might be some side effect
        if self.viewer:
            self.viewer.close()
            self.viewer = None
        self.backend.close()