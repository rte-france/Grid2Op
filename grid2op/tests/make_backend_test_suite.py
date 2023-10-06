# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import unittest
import re

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

def create_test_suite(make_backend_fun, add_name_cls, *args, tests_skipped=(), **kwargs):    
    def this_make_backend(self, detailed_infos_for_cascading_failures=False):
        return make_backend_fun(self,
                                *args,
                                detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures,
                                **kwargs)
    all_classes = []
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
        this_cls = type(f"{re.sub('Base', '', el.__name__)}_{add_name_cls}",
                        (el, unittest.TestCase),
                        {"make_backend": this_make_backend})
        all_classes.append(this_cls)
    
    return all_classes
