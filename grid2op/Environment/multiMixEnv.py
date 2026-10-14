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
from typing import Union, Optional

from grid2op.dtypes import dt_int
from grid2op.Space import (GridObjects,
                           RandomObject,
                           DEFAULT_N_BUSBAR_PER_SUB,
                           GRID2OP_CLASSES_ENV_FOLDER,
                           DEFAULT_ALLOW_DETACHMENT)
from grid2op.Exceptions import EnvError, Grid2OpException
from grid2op.Backend import Backend
from grid2op.Observation import BaseObservation
from grid2op.Environment.baseEnv import BaseEnv
from grid2op.typing_variables import RESET_OPTIONS_TYPING


class _OverloadNameMultiMixInfo:
    VALUE_ERROR_GETITEM : str = "You can only access member with integer and not with {}"
    
    def __init__(self,
                 path_cls=None,
                 path_env=None,
                 name_env=None,
                 add_to_name="",
                 mix_id=0,
                ):
        self.path_cls = path_cls
        self.path_env = path_env
        self.name_env = name_env
        self.add_to_name = add_to_name
        self.local_dir_tmpfolder = None
        self.mix_id = mix_id
        
    def __getitem__(self, arg):
        cls = type(self)
        try:
            arg_ = int(arg)
        except ValueError as exc_:
            raise ValueError(cls.VALUE_ERROR_GETITEM.format(type(arg))) from exc_
            
        if arg_ != arg:
            raise ValueError(cls.VALUE_ERROR_GETITEM.format(type(arg)))
        
        if arg_ < 0:
            # for stuff like "overload[-1]"
            arg_ += 6
            
        if arg_ == 0:
            return self.path_cls
        if arg_ == 1:
            return self.path_env
        if arg_ == 2:
            return self.name_env
        if arg_ == 3:
            return self.add_to_name
        if arg_ == 4:
            return self.local_dir_tmpfolder
        if arg_ == 5:
            return self.mix_id
        raise IndexError("_OverloadNameMultiMixInfo can only be used with index being 0, 1, 2, 3, 4 or 5")


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

    By default, each time you call the "reset" function a different mix is used. Mixes, by default
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
        allow_detachment=DEFAULT_ALLOW_DETACHMENT,
        _add_cls_nm_bk=True,
        _add_to_name="",  # internal, for test only, do not use !
        _compat_glop_version=None,  # internal, for test only, do not use !
        _test=False,
        **kwargs,
    ):
        GridObjects.__init__(self)
        RandomObject.__init__(self)
        self.current_env = None
        self.env_index = None
        self.mix_envs = {}
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
        self._ptr_backend_obj_first_env : Optional[Backend]= None
        _added_bk_name = ""
        
        if "backend" in kwargs:
            backendClass = type(kwargs["backend"])
            if hasattr(kwargs["backend"], "_my_kwargs"):
                # was introduced in grid2op 1.7.1
                backend_kwargs = kwargs["backend"]._my_kwargs
            _added_bk_name = kwargs["backend"].get_class_added_name()
            self._ptr_backend_obj_first_env = kwargs["backend"]
            del kwargs["backend"]
                        
        li_mix_nms = [mix_name for mix_name in sorted(os.listdir(envs_dir)) 
                      if (mix_name != GRID2OP_CLASSES_ENV_FOLDER and
                          mix_name != "__pycache__" and
                          os.path.isdir(os.path.join(envs_dir, mix_name)) 
                          )]
        if not li_mix_nms:
            raise EnvError("We did not find any mix in this multi-mix environment.")
        
        # Make sure GridObject class attributes are set from first env
        # Should be fine since the grid is the same for all envs
        self.multi_env_name = _OverloadNameMultiMixInfo(None, envs_dir, os.path.basename(os.path.abspath(envs_dir)), _add_to_name)

        env_for_init = self._aux_create_a_mix(envs_dir,
                                              li_mix_nms[0],
                                              True,  # first mix
                                              logger,
                                              backendClass,
                                              backend_kwargs,
                                              _add_cls_nm_bk,
                                              _add_to_name,
                                              _compat_glop_version,
                                              n_busbar,
                                              allow_detachment,
                                              _test,
                                              experimental_read_from_local_dir,
                                              self.multi_env_name,
                                              kwargs)    
        cls_res_me = self._aux_add_class_file(env_for_init)        
        if cls_res_me is not None:
            self.__class__ = cls_res_me
        else:
            self.__class__ = type(self).init_grid(type(env_for_init.backend), _local_dir_cls=env_for_init._local_dir_cls)
        self.mix_envs[li_mix_nms[0]] = env_for_init
        # TODO reuse same observation_space and action_space in all the envs maybe ?
        self.multi_env_name.path_cls = type(env_for_init)._PATH_GRID_CLASSES
        self.multi_env_name.name_env = env_for_init.env_name
        i = -1
        try:
            for i, mix_name in enumerate(li_mix_nms[1:]):
                mix_path = os.path.join(envs_dir, mix_name)
                if not os.path.isdir(mix_path):
                    continue
                mix = self._aux_create_a_mix(envs_dir,
                                             mix_name,
                                             False,  # first mix
                                             logger,
                                             backendClass,
                                             backend_kwargs,
                                             _add_cls_nm_bk,  # _add_cls_nm_bk already added in _add_to_name  ?
                                             _add_to_name,
                                             _compat_glop_version,
                                             n_busbar,
                                             allow_detachment,
                                             _test,
                                             experimental_read_from_local_dir,
                                             self.multi_env_name,
                                             kwargs)
                self.mix_envs[mix_name] = mix
        except Exception as exc_:
            err_msg = f"MultiMix environment creation failed at the creation of mix {mix_name} (mix {i+1+1} / {len(li_mix_nms)})"
            raise EnvError(err_msg) from exc_

        if len(self.mix_envs) == 0:
            err_msg = "MultiMix envs_dir did not contain any valid env"
            raise EnvError(err_msg)

        # tell every mix the "MultiMix" is responsible for deleting the 
        # folder that stores the classes definition
        for el in self.mix_envs.values():
            el._do_not_erase_local_dir_cls = True
        self.env_index = 0
        self.all_names = li_mix_nms
        self.current_env = self.mix_envs[self.all_names[self.env_index]]
        # legacy behaviour (using experimental_read_from_local_dir kwargs in env.make)
        if self._read_from_local_dir is not None:
            if os.path.split(self._read_from_local_dir)[1] == GRID2OP_CLASSES_ENV_FOLDER:
                self._do_not_erase_local_dir_cls = True
        else:
            self._do_not_erase_local_dir_cls = True
        
        # to prevent the cleaning of this tmp folder
        self.multi_env_name.local_dir_tmpfolder = None

    def _aux_aux_add_class_file(self, sys_path, env_for_init):
        # used for the old behaviour (setting experimental_read_from_local_dir=True in make)
        bk_type = type(env_for_init.backend)
        _PATH_GRID_CLASSES = bk_type._PATH_GRID_CLASSES
        cls_res_me = None
        try:
            bk_type._PATH_GRID_CLASSES = None
            my_type_tmp = MultiMixEnvironment.init_grid(gridobj=bk_type, _local_dir_cls=None)
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
        
    def _aux_add_class_file(self, env_for_init):
        # used for the "new" bahviour for grid2op make (automatic read from local dir)
        if env_for_init.classes_are_in_files() and env_for_init._local_dir_cls is not None:
            sys_path = os.path.abspath(env_for_init._local_dir_cls.name)
            self._local_dir_cls = env_for_init._local_dir_cls
            self.multi_env_name.local_dir_tmpfolder = self._local_dir_cls
            env_for_init._local_dir_cls = None
            # then generate the proper classes
            cls_res_me = self._aux_aux_add_class_file(sys_path, env_for_init)
            return cls_res_me
        return None
    
    def _aux_make_backend_from_cls(self, backendClass, backend_kwargs):
        # Special case for backend
        try:
            # should pass with grid2op >= 1.7.1
            bk = backendClass(**backend_kwargs)
        except TypeError as exc_:  # noqa: F841
            # with grid2Op version prior to 1.7.1
            # you might have trouble with 
            # "TypeError: __init__() got an unexpected keyword argument 'can_be_copied'"
            msg_ = ("Impossible to create a backend for each mix using the "
                    "backend key-word arguments. Falling back to creating "
                    "with no argument at all (default behaviour with grid2op <= 1.7.0).")
            warnings.warn(msg_)
            bk = backendClass()  
        return bk
    
    def _aux_create_a_mix(self,
                          envs_dir,
                          mix_name,
                          is_first_mix,
                          logger,
                          backendClass,
                          backend_kwargs,
                          _add_cls_nm_bk,
                          _add_to_name,
                          _compat_glop_version,
                          n_busbar,
                          allow_detachment,
                          _test,
                          experimental_read_from_local_dir,
                          multi_env_name : _OverloadNameMultiMixInfo,
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
        kwargs_make = dict(
            _add_cls_nm_bk=_add_cls_nm_bk,
            _add_to_name=_add_to_name,
            _compat_glop_version=_compat_glop_version,
            n_busbar=n_busbar,
            test=_test,
            logger=this_logger,
            experimental_read_from_local_dir=experimental_read_from_local_dir,
            _overload_name_multimix=multi_env_name,
            allow_detachment=allow_detachment,
            **kwargs)
        if is_first_mix:
            # in the first mix either I need to create the backend, or
            # pass the backend given in argument
            if self._ptr_backend_obj_first_env is not None:
                # I reuse the backend passed as object on the first mix
                bk = self._ptr_backend_obj_first_env
                kwargs_make["backend"] = bk
            elif backendClass is not None:
                # Special case for backend
                bk = self._aux_make_backend_from_cls(backendClass, backend_kwargs)
                kwargs_make["backend"] = bk             
        else:
            # in the other mixes, things are created with either a copy of the backend
            # or a new backend from the kwargs
            if self._ptr_backend_obj_first_env._can_be_copied:
                bk = self._ptr_backend_obj_first_env.copy()
                bk._is_loaded = False
            elif backendClass is not None:
                # Special case for backend
                bk = self._aux_make_backend_from_cls(self.mix_envs[self.all_names[0]]._raw_backend_class,
                                                     self._ptr_backend_obj_first_env._my_kwargs)
            kwargs_make["backend"] = bk
            
        mix = make(mix_path, **kwargs_make)
        mix.multimix_mix_name = mix_name
        multi_env_name.mix_id += 1
        if is_first_mix and self._ptr_backend_obj_first_env is None:
            # if the "backend" kwargs has not been provided in the user call to "make"
            # then I save a "pointer" to the backend of the first mix
            self._ptr_backend_obj_first_env = mix.backend
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
            r = self.mix_envs[self.all_names[self.env_index]]
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
        for mix in self.mix_envs.keys():
            yield mix

    def values(self):
        for mix in self.mix_envs.values():
            yield mix

    def items(self):
        for mix in self.mix_envs.items():
            yield mix

    def copy(self):
        if self.__closed:
            raise EnvError("This environment is closed, you cannot use it.")
        mix_envs = self.mix_envs
        self.mix_envs = None
        current_env = self.current_env
        self.current_env = None

        # do not copy these attributes
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
        res.mix_envs = {el: mix.copy() for el, mix in mix_envs.items()}
        res.current_env = res.mix_envs[res.all_names[res.env_index]]
        # finally deal with the ownership of the class folder
        res._local_dir_cls = _local_dir_cls
        res._do_not_erase_local_dir_cls = True
        
        # put back attributes of `self` that have been put aside
        self.mix_envs = mix_envs
        self.current_env = current_env
        self._local_dir_cls = _local_dir_cls
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
        return self.mix_envs[key]

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

        self.current_env = self.mix_envs[self.all_names[self.env_index]]
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
        except Exception as exc:  # noqa: F841
            raise Grid2OpException(
                "Cannot to seed with the seed provided."
                "Make sure it can be converted to a"
                "numpy 32 bits integer."
            ) from exc

        s = super().seed(seed)
        seeds = [s]
        max_dt_int = np.iinfo(dt_int).max
        for env in self.mix_envs.values():
            env_seed = self.space_prng.randint(max_dt_int)
            env_seeds = env.seed(env_seed)
            seeds.append(env_seeds)
        return seeds

    def set_chunk_size(self, new_chunk_size):
        if self.__closed:
            raise EnvError("This environment is closed, you cannot use it.")
        for mix in self.mix_envs.values():
            mix.set_chunk_size(new_chunk_size)

    def set_id(self, id_):
        if self.__closed:
            raise EnvError("This environment is closed, you cannot use it.")
        for mix in self.mix_envs.values():
            mix.set_id(id_)

    def deactivate_forecast(self):
        if self.__closed:
            raise EnvError("This environment is closed, you cannot use it.")
        for mix in self.mix_envs.values():
            mix.deactivate_forecast()

    def reactivate_forecast(self):
        if self.__closed:
            raise EnvError("This environment is closed, you cannot use it.")
        for mix in self.mix_envs.values():
            mix.reactivate_forecast()

    def set_thermal_limit(self, thermal_limit):
        """
        Set the thermal limit effectively.
        Will propagate to all underlying mixes
        """
        if self.__closed:
            raise EnvError("This environment is closed, you cannot use it.")
        for mix in self.mix_envs.values():
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

        for mix in self.mix_envs.values():
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
        for mix in self.mix_envs.values():
            mix.attach_layout(grid_layout)

    def __del__(self):
        """when the environment is garbage collected, free all the memory, including cross reference to itself in the observation space."""
        if not self.__closed:
            self.close()
            
    def generate_classes(self):
        mix_for_classes = self.mix_envs[self.all_names[0]]
        path_cls =  os.path.join(self.multi_env_name.path_env, GRID2OP_CLASSES_ENV_FOLDER)
        if not os.path.exists(path_cls):
            try:
                os.mkdir(path_cls)
            except FileExistsError:
                pass
        mix_for_classes.generate_classes(sys_path=path_cls)
        self._aux_aux_add_class_file(path_cls, mix_for_classes)
