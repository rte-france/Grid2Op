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

def _make_and_add_cls(el, add_name_cls, this_make_backend, add_to_module, all_classes):
        this_cls = type(f"{re.sub('Base', '', el.__name__)}_{add_name_cls}",
                        (el, unittest.TestCase),
                        {"make_backend": this_make_backend})
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

    And then run this script normally `python my_script.py` and observe the results
    
    .. warning::
        Do not forget to include the `self` as the first argument of the `this_make_backend` (or whatever 
        you decide to name it) even if you don't use it ! 
        
        Otherwise this script will NOT work !
        
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
    all_classes = []
    _make_and_add_cls(AAATestBackendAPI, add_name_cls, this_make_backend, add_to_module, all_classes)
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
        _make_and_add_cls(el, add_name_cls, this_make_backend, add_to_module, all_classes)
    
    return all_classes
