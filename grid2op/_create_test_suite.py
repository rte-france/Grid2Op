# Copyright (c) 2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import unittest
import re
import sys

from grid2op.tests.aaa_test_backend_interface import AAATestBackendAPI
from grid2op.tests.BaseBackendTest import (BaseTestNames, BaseTestLoadingCase, BaseTestLoadingBackendFunc,
                                           BaseTestTopoAction, BaseTestEnvPerformsCorrectCascadingFailures,
                                           BaseTestChangeBusAffectRightBus, BaseTestShuntAction,
                                           BaseTestResetEqualsLoadGrid, BaseTestVoltageOWhenDisco, BaseTestChangeBusSlack,
                                           BaseIssuesTest, BaseStatusActions,
                                           BaseTestStorageAction)
from grid2op.tests.test_Environment import (BaseTestLoadingBackendPandaPower,
                                            BaseTestResetOk,
                                            BaseTestResetAfterCascadingFailure,
                                            BaseTestCascadingFailure)
from grid2op.tests.BaseRedispTest import (BaseTestRedispatch, BaseTestRedispatchChangeNothingEnvironment,
                                          BaseTestRedispTooLowHigh, BaseTestDispatchRampingIllegalETC,
                                          BaseTestLoadingAcceptAlmostZeroSumRedisp)


# Issue131Tester

def _make_and_add_cls(el,
                      add_name_cls,
                      this_make_backend,
                      add_to_module,
                      all_classes,
                      get_paths,
                      get_casefiles):
        this_methods =  {"make_backend": this_make_backend}
        if get_paths is not None:
            if el.__name__ in get_paths:
                this_methods["get_path"] = get_paths[el.__name__]
        if get_casefiles is not None:
            if el.__name__ in get_casefiles:
                this_methods["get_casefile"] = get_casefiles[el.__name__]
        this_cls = type(f"{re.sub('Base', '', el.__name__)}_{add_name_cls}",
                        (el, unittest.TestCase),
                        this_methods)
        if add_to_module is not None:
            # make the created class visible to the default module
            setattr(sys.modules[add_to_module],
                    this_cls.__name__,
                    this_cls,
                    )
        all_classes.append(this_cls)
        

def create_test_suite(make_backend_fun,
                      add_name_cls,
                      *args,
                      add_to_module=None,
                      extended_test=True,
                      tests_skipped=(),
                      get_paths=None,
                      get_casefiles=None,
                      **kwargs):    
    """This function helps you create a test suite to test the behaviour of your agent.
    
    We recommend to use it this way:
    
    First use `extended_test=False` and make sure all the tests pass (or at least understand why
    the failing tests are failing)
    
    Then, once your backend is fully working, you can pass the kwarg `extended_test=False` to 
    test it more in depth.
    
    .. alert::
        You need to install grid2op from source (from github), in developer mode to use this function !
        
        For example you can do (inside a virtual env):
        
        - git clone https://github.com/rte-france/grid2op.git grid2op_dev
        - cd grid2op_dev 
        - pip install -e .
        
        And then you can use this function. It will **NOT** work if you install grid2op from pypi (with pip install grid2op)
        or if you install grid2op "not in editable mode".
        
    .. note::
        If pip complains at the installation, then you can remove the "pyproject.toml" and re run `pip install -e .`

    The best way to use it is (in my humble opinion) is first to make a script (*eg* `my_tests.py`) like this:
    
    .. code-block:: python
    
        from grid2op.create_test_suite import create_test_suite
        
        def this_make_backend(self, detailed_infos_for_cascading_failures=False):
            # of course replace this function by the function that creates your backend !
            # this should be the only line you have to change in this script
            return PandaPowerBackend(
                    detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures
                )
        add_name_cls = "test_functionality"  # you can change this if you want to
        
        create_test_suite(make_backend_fun=this_make_backend,
                          add_name_cls=add_name_cls,
                          add_to_module=__name__,
                          extended_test=False)
                          
        if __name__ == "__main__":
            unittest.main()
            
    .. warning::
        Do not forget to include the `self` as the first argument of the `this_make_backend` (or whatever 
        you decide to name it) even if you don't use it ! 
        
        Otherwise this script will NOT work !
        
    And then run this script normally `python my_script.py` and observe the results
    
    
    You can also import it this way (a bit more verbose but might give you more control on which tests is launched)
    
    .. code-block:: python
    
        import unittest
        from grid2op.tests.aaa_test_backend_interface import AAATestBackendAPI
        
        class TestBackendAPI_PyPoBk(AAATestBackendAPI, unittest.TestCase):
            
            def make_backend(self, detailed_infos_for_cascading_failures=False):
                # replace this with the backend you want to test. It should return a backend !
                return  LightSimBackend(loader_method="pypowsybl",
                                        loader_kwargs=_aux_get_loader_kwargs(),
                                        detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures)
                                        
    It allows also to modify the grid format (for example) that your backend can read from:

    .. code-block:: python
    
        import unittest
        from grid2op.tests.aaa_test_backend_interface import AAATestBackendAPI
        
        class TestBackendAPI_PyPoBk(AAATestBackendAPI, unittest.TestCase):
            def get_path(self):
                # if you want to change the path from which data will be read
                return path_case_14_storage_iidm
            
            def get_casefile(self):
                # if you want to change the grid file that will be read by the backend.
                return "grid.xiidm"
            
            def make_backend(self, detailed_infos_for_cascading_failures=False):
                # replace this with the backend you want to test. It should return a backend !
                return  LightSimBackend(loader_method="pypowsybl",
                                        loader_kwargs=_aux_get_loader_kwargs(),
                                        detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures)
    
    Yet another use, if you want more customization:
    
    .. code-block:: python
    
        def get_path_test_api(self):
            return path
        
        def get_casefile(self):
            return "grid.xiidm"
        
        res = create_test_suite(make_backend_fun=this_make_backend,
                                add_name_cls=add_name_cls,
                                add_to_module=__name__,
                                extended_test=False,  # for now keep `extended_test=False` until all problems are solved
                                get_paths={"AAATestBackendAPI": get_path_test_api},
                                get_casefiles={"AAATestBackendAPI": get_casefile}
                                )         
    
    Parameters
    ----------
    make_backend_fun : _type_
        _description_
    add_name_cls : _type_
        _description_
    add_to_module : _type_, optional
        _description_, by default None
    extended_test : bool, optional
        _description_, by default True
    tests_skipped : tuple, optional
        # TODO !
        _description_, by default ()
    """
    
    def this_make_backend(self, detailed_infos_for_cascading_failures=False):
        return make_backend_fun(self,
                                *args,
                                detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures,
                                **kwargs)
    if get_paths is None:
        get_paths = {}
    if get_casefiles is None:
        get_casefiles = {}
        
    all_classes = []
    _make_and_add_cls(AAATestBackendAPI, add_name_cls, this_make_backend, add_to_module, all_classes, get_paths, get_casefiles)
    if not extended_test:
        return
    
    # Does only the last one...
    for el in [BaseTestNames,
               BaseTestLoadingCase,
               BaseTestLoadingBackendFunc,
               BaseTestTopoAction,
               BaseTestEnvPerformsCorrectCascadingFailures,
               BaseTestChangeBusAffectRightBus,
               BaseTestShuntAction,
               BaseTestResetEqualsLoadGrid,
               BaseTestVoltageOWhenDisco,
               BaseTestChangeBusSlack,
               BaseIssuesTest,
               BaseStatusActions,
               BaseTestStorageAction,
               BaseTestLoadingBackendPandaPower,
               BaseTestResetOk,
               BaseTestResetAfterCascadingFailure,
               BaseTestCascadingFailure,
               BaseTestRedispatch,
               BaseTestRedispatchChangeNothingEnvironment,
               BaseTestRedispTooLowHigh,
               BaseTestDispatchRampingIllegalETC,
               BaseTestLoadingAcceptAlmostZeroSumRedisp]:
        _make_and_add_cls(el, add_name_cls, this_make_backend, add_to_module, all_classes, get_paths, get_casefiles)
    
    return all_classes
