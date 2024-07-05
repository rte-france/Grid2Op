# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import os
import warnings
import numpy as np
import copy
from typing import Any, Dict, Tuple, Union, List, Literal

from grid2op.dtypes import dt_int, dt_float
from grid2op.Space import GridObjects, RandomObject, DEFAULT_N_BUSBAR_PER_SUB
from grid2op.Exceptions import EnvError, Grid2OpException
from grid2op.Observation import BaseObservation
from grid2op.MakeEnv.PathUtils import USE_CLASS_IN_FILE
from grid2op.Environment.baseEnv import BaseEnv
from grid2op.typing_variables import STEP_INFO_TYPING, RESET_OPTIONS_TYPING


class MultiMixEnvironment(GridObjects, RandomObject):
    """
    This class represent a single powergrid configuration,
    backed by multiple environments parameters and chronics

    It implements most of the :class:`BaseEnv` public interface:
    so it can be used as a more classic environment.

    MultiMixEnvironment environments behave like a superset of the environment: they
    are made of sub environments (called mixes) that are grid2op regular :class:`Environment`.
    You might think the MultiMixEnvironment as a dictionary of :class:`Environment` that implements
    some of the :class:`BaseEnv` interface such as :func:`BaseEnv.step` or :func:`BaseEnv.reset`.

    By default, each time you call the "step" function a different mix is used. Mixes, by default
    are looped through always in the same order. You can see the Examples section for information
    about control of these


    Examples
    --------
    In this section we present some common use of the MultiMix environment.

    **Basic Usage**

    You can think of a MultiMixEnvironment as any :class:`Environment`. So this is a perfectly
    valid way to use a MultiMix:

    .. code-block:: python

        import grid2op
        from grid2op.Agent import RandomAgent

        # we use an example of a multimix dataset attached with grid2op pacakage
        multimix_env = grid2op.make("l2rpn_neurips_2020_track2", test=True)

        # define an agent like in any environment
        agent = RandomAgent(multimix_env.action_space)

        # and now you can do the open ai gym loop
        NB_EPISODE = 10
        for i in range(NB_EPISODE):
            obs = multimix_env.reset()
            # each time "reset" is called, another mix is used.
            reward = multimix_env.reward_range[0]
            done = False
            while not done:
                act = agent.act(obs, reward, done)
                obs, reward, done, info = multimix_env.step(act)

    **Use each mix one after the other**

    In case you want to study each mix independently, you can iterate through the MultiMix
    in a pythonic way. This makes it easy to perform, for example, 10 episode for a given mix
    before passing to the next one.

    .. code-block:: python

        import grid2op
        from grid2op.Agent import RandomAgent

        # we use an example of a multimix dataset attached with grid2op pacakage
        multimix_env = grid2op.make("l2rpn_neurips_2020_track2", test=True)

        NB_EPISODE = 10
        for mix in multimix_env:
            # mix is a regular environment, you can do whatever you want with it
            # for example
            for i in range(NB_EPISODE):
                obs = multimix_env.reset()
                # each time "reset" is called, another mix is used.
                reward = multimix_env.reward_range[0]
                done = False
                while not done:
                    act = agent.act(obs, reward, done)
                    obs, reward, done, info = multimix_env.step(act)


    **Selecting a given Mix**

    Sometimes it might be interesting to study only a given mix.
    For that you can use the `[]` operator to select only a given mix (which is a grid2op environment)
    and use it as you would.

    This can be done with:

    .. code-block:: python

        import grid2op
        from grid2op.Agent import RandomAgent

        # we use an example of a multimix dataset attached with grid2op pacakage
        multimix_env = grid2op.make("l2rpn_neurips_2020_track2", test=True)

        # define an agent like in any environment
        agent = RandomAgent(multimix_env.action_space)

        # list all available mixes:
        mixes_names = list(multimix_env.keys())

        # and now supposes we want to study only the first one
        mix = multimix_env[mixes_names[0]]

        # and now you can do the open ai gym loop, or anything you want with it
        NB_EPISODE = 10
        for i in range(NB_EPISODE):
            obs = mix.reset()
            # each time "reset" is called, another mix is used.
            reward = mix.reward_range[0]
            done = False
            while not done:
                act = agent.act(obs, reward, done)
                obs, reward, done, info = mix.step(act)

    **Using the Runner**

    For MultiMixEnvironment using the :class:`grid2op.Runner.Runner` cannot be done in a
    straightforward manner. Here we give an example on how to do it.

    .. code-block:: python

        import os
        import grid2op
        from grid2op.Agent import RandomAgent

        # we use an example of a multimix dataset attached with grid2op pacakage
        multimix_env = grid2op.make("l2rpn_neurips_2020_track2", test=True)

        # you can use the runner as following
        PATH = "PATH/WHERE/YOU/WANT/TO/SAVE/THE/RESULTS"
        for mix in multimix_env:
            runner = Runner(**mix.get_params_for_runner(), agentClass=RandomAgent)
            runner.run(nb_episode=1,
                       path_save=os.path.join(PATH,mix.name))

    """

    KEYS_RESET_OPTIONS = BaseEnv.KEYS_RESET_OPTIONS
    
    def __init__(
        self,
        envs_dir,
        logger=None,
        experimental_read_from_local_dir=None,
        n_busbar=DEFAULT_N_BUSBAR_PER_SUB,
        _add_to_name="",  # internal, for test only, do not use !
        _compat_glop_version=None,  # internal, for test only, do not use !
        _test=False,
        **kwargs,
    ):
        GridObjects.__init__(self)
        RandomObject.__init__(self)
        self.current_env = None
        self.env_index = None
        self.mix_envs = []
        self._env_dir = os.path.abspath(envs_dir)
        self.__closed = False
        self._do_not_erase_local_dir_cls = False
        self._local_dir_cls = None
        if not os.path.exists(envs_dir):
            raise EnvError(f"There is nothing at {envs_dir}")
        # Special case handling for backend
        # TODO: with backend.copy() instead !
        backendClass = None
        backend_kwargs = {}
        if "backend" in kwargs:
            backendClass = type(kwargs["backend"])
            if hasattr(kwargs["backend"], "_my_kwargs"):
                # was introduced in grid2op 1.7.1
                backend_kwargs = kwargs["backend"]._my_kwargs
            del kwargs["backend"]
        
        li_mix_nms = [mix_name for mix_name in sorted(os.listdir(envs_dir)) if os.path.isdir(os.path.join(envs_dir, mix_name))]
        if not li_mix_nms:
            raise EnvError("We did not find any mix in this multi-mix environment.")
        
        # Make sure GridObject class attributes are set from first env
        # Should be fine since the grid is the same for all envs
        multi_env_name = (envs_dir, os.path.basename(os.path.abspath(envs_dir)), _add_to_name)
        env_for_init = self._aux_create_a_mix(envs_dir,
                                              li_mix_nms[0],
                                              logger,
                                              backendClass,
                                              backend_kwargs,
                                              _add_to_name,
                                              _compat_glop_version,
                                              n_busbar,
                                              _test,
                                              experimental_read_from_local_dir,
                                              multi_env_name,
                                              kwargs)
        cls_res_me = self._aux_add_class_file(env_for_init)
        if cls_res_me is not None:
            self.__class__ = cls_res_me
        else:
            self.__class__ = type(self).init_grid(type(env_for_init.backend), _local_dir_cls=env_for_init._local_dir_cls)
        self.mix_envs.append(env_for_init)
        self._local_dir_cls = env_for_init._local_dir_cls
        
        # TODO reuse same observation_space and action_space in all the envs maybe ?
        try:
            for mix_name in li_mix_nms[1:]:
                mix_path = os.path.join(envs_dir, mix_name)
                if not os.path.isdir(mix_path):
                    continue
                mix = self._aux_create_a_mix(envs_dir,
                                             mix_name,
                                             logger,
                                             backendClass,
                                             backend_kwargs,
                                             _add_to_name,
                                             _compat_glop_version,
                                             n_busbar,
                                             _test,
                                             experimental_read_from_local_dir,
                                             multi_env_name,
                                             kwargs)
                self.mix_envs.append(mix)
        except Exception as exc_:
            err_msg = "MultiMix environment creation failed: {}".format(exc_)
            raise EnvError(err_msg) from exc_

        if len(self.mix_envs) == 0:
            err_msg = "MultiMix envs_dir did not contain any valid env"
            raise EnvError(err_msg)

        # tell every mix the "MultiMix" is responsible for deleting the 
        # folder that stores the classes definition
        for el in self.mix_envs:
            el._do_not_erase_local_dir_cls = True
        self.env_index = 0
        self.current_env = self.mix_envs[self.env_index]

    def _aux_add_class_file(self, env_for_init):
        if env_for_init.classes_are_in_files() and env_for_init._local_dir_cls is not None:
            bk_type = type(env_for_init.backend)
            sys_path = os.path.abspath(env_for_init._local_dir_cls.name)
            self._local_dir_cls = env_for_init._local_dir_cls
            env_for_init._local_dir_cls = None
            # then generate the proper classes
            _PATH_GRID_CLASSES = bk_type._PATH_GRID_CLASSES
            try:
                bk_type._PATH_GRID_CLASSES = None
                my_type_tmp = type(self).init_grid(gridobj=bk_type, _local_dir_cls=None)
                txt_, cls_res_me = BaseEnv._aux_gen_classes(my_type_tmp,
                                                            sys_path,
                                                            _add_class_output=True)
                # then add the class to the init file
                with open(os.path.join(sys_path, "__init__.py"), "a", encoding="utf-8") as f:
                    f.write(txt_)
            finally:
                # make sure to put back the correct _PATH_GRID_CLASSES
                bk_type._PATH_GRID_CLASSES = _PATH_GRID_CLASSES
            return cls_res_me
        return None
        
    def _aux_create_a_mix(self,
                          envs_dir,
                          mix_name,
                          logger,
                          backendClass,
                          backend_kwargs,
                          _add_to_name,
                          _compat_glop_version,
                          n_busbar,
                          _test,
                          experimental_read_from_local_dir,
                          multi_env_name,
                          kwargs
                          ):
        # Inline import to prevent cyclical import
        from grid2op.MakeEnv.Make import make
        
        this_logger = (
            logger.getChild(f"MultiMixEnvironment_{mix_name}")
            if logger is not None
            else None
        )
        mix_path = os.path.join(envs_dir, mix_name)
        # Special case for backend
        if backendClass is not None:
            try:
                # should pass with grid2op >= 1.7.1
                bk = backendClass(**backend_kwargs)
            except TypeError as exc_:
                # with grid2Op version prior to 1.7.1
                # you might have trouble with 
                # "TypeError: __init__() got an unexpected keyword argument 'can_be_copied'"
                msg_ = ("Impossible to create a backend for each mix using the "
                        "backend key-word arguments. Falling back to creating "
                        "with no argument at all (default behaviour with grid2op <= 1.7.0).")
                warnings.warn(msg_)
                bk = backendClass()
            mix = make(
                mix_path,
                backend=bk,
                _add_to_name=_add_to_name,
                _compat_glop_version=_compat_glop_version,
                n_busbar=n_busbar,
                test=_test,
                logger=this_logger,
                experimental_read_from_local_dir=experimental_read_from_local_dir,
                _overload_name_multimix=multi_env_name,
                **kwargs,
            )
        else:
            mix = make(
                mix_path,
                n_busbar=n_busbar,
                _add_to_name=_add_to_name,
                _compat_glop_version=_compat_glop_version,
                test=_test,
                logger=this_logger,
                experimental_read_from_local_dir=experimental_read_from_local_dir,
                _overload_name_multimix=multi_env_name,
                **kwargs,
            )
        return mix
    
    def get_path_env(self):
        """
        Get the path that allows to create this environment.

        It can be used for example in `grid2op.utils.underlying_statistics` to save the information directly inside
        the environment data.

        """
        if self.__closed:
            raise EnvError("This environment is closed, you cannot use it.")
        return self._env_dir

    @property
    def current_index(self):
        return self.env_index

    def __len__(self):
        return len(self.mix_envs)

    def __iter__(self):
        """
        Operator __iter__ overload to make a ``MultiMixEnvironment`` iterable

        .. code-block:: python

            import grid2op
            from grid2op.Environment import MultiMixEnvironment
            from grid2op.Runner import Runner

            mm_env = MultiMixEnvironment("/path/to/multi/dataset/folder")

            for env in mm_env:
                run_p = env.get_params_for_runner()
                runner = Runner(**run_p)
                runner.run(nb_episode=1, max_iter=-1)
        """
        self.env_index = 0
        return self

    def __next__(self):
        if self.env_index < len(self.mix_envs):
            r = self.mix_envs[self.env_index]
            self.env_index = self.env_index + 1
            return r
        else:
            self.env_index = 0
            raise StopIteration

    def __getattr__(self, name):
        # TODO what if name is an integer ? make it possible to loop with integer here
        if self.__closed:
            raise EnvError("This environment is closed, you cannot use it.")
        return getattr(self.current_env, name)

    def keys(self):
        for mix in self.mix_envs:
            yield mix.multimix_mix_name

    def values(self):
        for mix in self.mix_envs:
            yield mix

    def items(self):
        for mix in self.mix_envs:
            yield mix.multimix_mix_name, mix

    def copy(self):
        if self.__closed:
            raise EnvError("This environment is closed, you cannot use it.")
        mix_envs = self.mix_envs
        self.mix_envs = None
        current_env = self.current_env
        self.current_env = None

        # do not copy these attributes
        _do_not_erase_local_dir_cls = self._do_not_erase_local_dir_cls
        self._do_not_erase_local_dir_cls = None
        _local_dir_cls = self._local_dir_cls
        self._local_dir_cls = None
        
        # create the new object and copy the normal attribute
        cls = self.__class__
        res = cls.__new__(cls)
        for k in self.__dict__:
            if k == "mix_envs" or k == "current_env":
                # this is handled elsewhere
                continue
            setattr(res, k, copy.deepcopy(getattr(self, k)))
        # now deal with the mixes
        res.mix_envs = [mix.copy() for mix in mix_envs]
        res.current_env = res.mix_envs[res.env_index]
        # finally deal with the ownership of the class folder
        res._do_not_erase_local_dir_cls = True
        res._local_dir_cls = _local_dir_cls
        
        # put back attributes of `self` that have been put aside
        self.mix_envs = mix_envs
        self.current_env = current_env
        self._local_dir_cls = _local_dir_cls
        self._do_not_erase_local_dir_cls = _do_not_erase_local_dir_cls
        
        return res

    def __getitem__(self, key):
        """
        Operator [] overload for accessing underlying mixes by name

        .. code-block:: python

            import grid2op
            from grid2op.Environment import MultiMixEnvironment

            mm_env = MultiMixEnvironment("/path/to/multi/dataset/folder")

            mix1_env.name = mm_env["mix_1"]
            assert mix1_env == "mix_1"
            mix2_env.name = mm_env["mix_2"]
            assert mix2_env == "mix_2"
        """
        if self.__closed:
            raise EnvError("This environment is closed, you cannot use it.")
        # Search for key
        for mix in self.mix_envs:
            if mix.multimix_mix_name == key:
                return mix

        # Not found by name
        raise KeyError

    def reset(self, 
              *,
              seed: Union[int, None] = None,
              random=False,
              options: RESET_OPTIONS_TYPING = None) -> BaseObservation:
        
        if self.__closed:
            raise EnvError("This environment is closed, you cannot use it.")
        
        if options is not None:
            for el in options:
                if el not in type(self).KEYS_RESET_OPTIONS:
                    raise EnvError(f"You tried to customize the `reset` call with some "
                                   f"`options` using the key `{el}` which is invalid. "
                                   f"Only keys in {sorted(list(type(self).KEYS_RESET_OPTIONS))} "
                                   f"can be used.")
                    
        if random:
            self.env_index = self.space_prng.randint(len(self.mix_envs))
        else:
            self.env_index = (self.env_index + 1) % len(self.mix_envs)

        self.current_env = self.mix_envs[self.env_index]
        return self.current_env.reset(seed=seed, options=options)

    def seed(self, seed=None):
        """
        Set the seed of this :class:`Environment` for a better control
        and to ease reproducible experiments.

        Parameters
        ----------
        seed: ``int``
           The seed to set.

        Returns
        ---------
        seeds: ``list``
            The seed used to set the prng (pseudo random number generator)
            for all environments, and each environment ``tuple`` seeds

        """
        if self.__closed:
            raise EnvError("This environment is closed, you cannot use it.")
        try:
            seed = np.array(seed).astype(dt_int)
        except Exception as e:
            raise Grid2OpException(
                "Cannot to seed with the seed provided."
                "Make sure it can be converted to a"
                "numpy 32 bits integer."
            )

        s = super().seed(seed)
        seeds = [s]
        max_dt_int = np.iinfo(dt_int).max
        for env in self.mix_envs:
            env_seed = self.space_prng.randint(max_dt_int)
            env_seeds = env.seed(env_seed)
            seeds.append(env_seeds)
        return seeds

    def set_chunk_size(self, new_chunk_size):
        if self.__closed:
            raise EnvError("This environment is closed, you cannot use it.")
        for mix in self.mix_envs:
            mix.set_chunk_size(new_chunk_size)

    def set_id(self, id_):
        if self.__closed:
            raise EnvError("This environment is closed, you cannot use it.")
        for mix in self.mix_envs:
            mix.set_id(id_)

    def deactivate_forecast(self):
        if self.__closed:
            raise EnvError("This environment is closed, you cannot use it.")
        for mix in self.mix_envs:
            mix.deactivate_forecast()

    def reactivate_forecast(self):
        if self.__closed:
            raise EnvError("This environment is closed, you cannot use it.")
        for mix in self.mix_envs:
            mix.reactivate_forecast()

    def set_thermal_limit(self, thermal_limit):
        """
        Set the thermal limit effectively.
        Will propagate to all underlying mixes
        """
        if self.__closed:
            raise EnvError("This environment is closed, you cannot use it.")
        for mix in self.mix_envs:
            mix.set_thermal_limit(thermal_limit)

    def __enter__(self):
        """
        Support *with-statement* for the environment.

        """
        return self

    def __exit__(self, *args):
        """
        Support *with-statement* for the environment.

        """
        self.close()
        # propagate exception
        return False

    def close(self):
        if self.__closed:
            return

        for mix in self.mix_envs:
            mix.close()
            
        self.__closed = True
        
        # free the resources (temporary directory)
        if self._do_not_erase_local_dir_cls:
            # The resources are not held by this env, so 
            # I do not remove them
            # (case for ObsEnv or ForecastedEnv)
            return
        BaseEnv._aux_close_local_dir_cls(self)
            

    def attach_layout(self, grid_layout):
        if self.__closed:
            raise EnvError("This environment is closed, you cannot use it.")
        for mix in self.mix_envs:
            mix.attach_layout(grid_layout)

    def __del__(self):
        """when the environment is garbage collected, free all the memory, including cross reference to itself in the observation space."""
        if not self.__closed:
            self.close()
            
    def generate_classes(self):
        # TODO this is not really a good idea, as the multi-mix itself is not read from the
        # files !
        # for mix in self.mix_envs:
            # mix.generate_classes()
        mix_for_classes = self.mix_envs[0]
        path_cls = os.path.join(mix_for_classes.get_path_env(), "_grid2op_classes")
        if not os.path.exists(path_cls):
            try:
                os.mkdir(path_cls)
            except FileExistsError:
                pass
        mix_for_classes.generate_classes()
        self._aux_add_class_file(mix_for_classes)
