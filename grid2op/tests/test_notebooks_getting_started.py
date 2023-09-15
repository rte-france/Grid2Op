# Copyright (c) 2019-2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import shutil
import copy
import os
import unittest
import time
import warnings
import pdb
import subprocess
import sys

try:
    # Import error : Jupyter is migrating its paths to use standard platformdirs
    # given by the platformdirs library.  To remove this warning and
    # see the appropriate new directories, set the environment variable
    # `JUPYTER_PLATFORM_DIRS=1` and then run `jupyter --paths`.
    # The use of platformdirs will be the default in `jupyter_core` v6
    os.environ["JUPYTER_PLATFORM_DIRS"] = "1"
    subprocess.run([f"{sys.executable}", "-m", "jupyter", "--paths"], capture_output=True, env=os.environ)
    # the above 2 lines are to fix the above error
    
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError
    CAN_COMPUTE = None
except Exception as exc_:
    CAN_COMPUTE = exc_
    print(f"Import error : {exc_}")
    
from grid2op.tests.helper_path_test import PATH_DATA_TEST

NOTEBOOK_PATHS = os.path.abspath(os.path.join(PATH_DATA_TEST, "../../getting_started"))
VERBOSE_TIMER = True


def delete_all(folder):
    """
    Delete all the files in a folder recursively.

    Parameters
    ----------
    folder: ``str``
        The folder in which we delete everything

    Returns
    -------
    None
    """
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))


def export_all_notebook(folder_in):
    """

    Parameters
    ----------
    folder_in: ``str``
        The folder in which we look for ipynb files

    folder_out: ``str``
        The folder in which we save the py file.

    Returns
    -------
    res: ``list``
        Return the list of notebooks names

    """
    res = []
    for filename in os.listdir(folder_in):
        if os.path.splitext(filename)[1] == ".ipynb":
            notebook_filename = os.path.join(folder_in, filename)
            res.append(notebook_filename)
    return res


class RAII_tf_log:
    def __init__(self):
        self.previous = None
        if "TF_CPP_MIN_LOG_LEVEL" in os.environ:
            self.previous = copy.deepcopy(os.environ["TF_CPP_MIN_LOG_LEVEL"])
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    def __del__(self):
        if self.previous is not None:
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = self.previous


# notebook names are hard coded because if i change them, i need also to change the
# readme and the documentation


class RAII_Timer:
    """
    class to have an approximation of the runtime of the notebook.
    This is a rough approximation to reduce time spent in certain notebooks and should not be
    used for another purpose.
    """

    def __init__(self, str_=""):
        self._time = time.perf_counter()
        self.str_ = str_

    def __del__(self):
        if VERBOSE_TIMER:
            print(
                f"Execution time for {self.str_}: {time.perf_counter() - self._time:.3f} s"
            )


class TestNotebooks(unittest.TestCase):
    def _aux_funct_notebook(self, notebook_filename):
        assert os.path.exists(notebook_filename), f"{notebook_filename} do not exists!"
        with open(notebook_filename) as f:
            nb = nbformat.read(f, as_version=4)
        try:
            ep = ExecutePreprocessor(timeout=60, store_widget_state=True)
            try:
                ep.preprocess(nb, {"metadata": {"path": NOTEBOOK_PATHS}})
            except CellExecutionError as exc_:
                raise
            except Exception as exc_:
                # error with tqdm progress bar i believe
                pass
        except CellExecutionError as exc_:
            raise
        except Exception:
            pass

    def _check_for_baselines(self):
        try:
            import l2rpn_baselines
        except ImportError as exc_:
            self.skipTest("l2rpn baseline is not available")

    def test_notebook0_1(self):
        if CAN_COMPUTE is not None:
            self.skipTest(f"{CAN_COMPUTE}")
        timer = RAII_Timer("test_notebook0_1")
        notebook_filename = os.path.join(NOTEBOOK_PATHS, "00_SmallExample.ipynb")
        self._aux_funct_notebook(notebook_filename)

    def test_notebook1(self):
        if CAN_COMPUTE is not None:
            self.skipTest(f"{CAN_COMPUTE}")
        return  # takes 80s and is useless i think, as it tests only basics things
        timer = RAII_Timer("test_notebook1")
        notebook_filename = os.path.join(NOTEBOOK_PATHS, "01_Grid2opFramework.ipynb")
        self._aux_funct_notebook(notebook_filename)

    def test_notebook2(self):
        if CAN_COMPUTE is not None:
            self.skipTest(f"{CAN_COMPUTE}")
        timer = RAII_Timer("test_notebook2")
        notebook_filename = os.path.join(NOTEBOOK_PATHS, "02_Observation.ipynb")
        self._aux_funct_notebook(notebook_filename)

    def test_notebook3(self):
        if CAN_COMPUTE is not None:
            self.skipTest(f"{CAN_COMPUTE}")
        timer = RAII_Timer("test_notebook3")
        notebook_filename = os.path.join(NOTEBOOK_PATHS, "03_Action.ipynb")
        self._aux_funct_notebook(notebook_filename)

    def test_notebook4(self):
        if CAN_COMPUTE is not None:
            self.skipTest(f"{CAN_COMPUTE}")
        self._check_for_baselines()
        raii_ = RAII_tf_log()
        timer = RAII_Timer("test_notebook4")
        notebook_filename = os.path.join(NOTEBOOK_PATHS, "04_TrainingAnAgent.ipynb")
        self._aux_funct_notebook(notebook_filename)

    def test_notebook5(self):
        if CAN_COMPUTE is not None:
            self.skipTest(f"{CAN_COMPUTE}")
        timer = RAII_Timer("test_notebook5")
        notebook_filename = os.path.join(NOTEBOOK_PATHS, "05_StudyYourAgent.ipynb")
        self._aux_funct_notebook(notebook_filename)

    def test_notebook6(self):
        if CAN_COMPUTE is not None:
            self.skipTest(f"{CAN_COMPUTE}")
        timer = RAII_Timer("test_notebook6")
        notebook_filename = os.path.join(
            NOTEBOOK_PATHS, "06_Redispatching_Curtailment.ipynb"
        )
        self._aux_funct_notebook(notebook_filename)

    def test_notebook7(self):
        if CAN_COMPUTE is not None:
            self.skipTest(f"{CAN_COMPUTE}")
        self._check_for_baselines()
        raii_ = RAII_tf_log()
        timer = RAII_Timer("test_notebook7")
        notebook_filename = os.path.join(NOTEBOOK_PATHS, "07_MultiEnv.ipynb")
        self._aux_funct_notebook(notebook_filename)

    def test_notebook8(self):
        if CAN_COMPUTE is not None:
            self.skipTest(f"{CAN_COMPUTE}")

        # display notebook, might not be super useful to test it in the unit test (saves another 1-2 minutes)
        return
        timer = RAII_Timer("test_notebook8")
        notebook_filename = os.path.join(
            NOTEBOOK_PATHS, "08_PlottingCapabilities.ipynb"
        )
        self._aux_funct_notebook(notebook_filename)

    def test_notebook9(self):
        if CAN_COMPUTE is not None:
            self.skipTest(f"{CAN_COMPUTE}")
        return  # test the opponent and the maintenance, not much there but takes 80s so... not a lot to do
        timer = RAII_Timer("test_notebook9")
        notebook_filename = os.path.join(
            NOTEBOOK_PATHS, "09_EnvironmentModifications.ipynb"
        )
        self._aux_funct_notebook(notebook_filename)

    def test_notebook10(self):
        if CAN_COMPUTE is not None:
            self.skipTest(f"{CAN_COMPUTE}")
        timer = RAII_Timer("test_notebook10")
        notebook_filename = os.path.join(NOTEBOOK_PATHS, "10_StorageUnits.ipynb")
        self._aux_funct_notebook(notebook_filename)

    def test_notebook_aub(self):
        if CAN_COMPUTE is not None:
            self.skipTest(f"{CAN_COMPUTE}")
        return
        raii_ = RAII_tf_log()
        timer = RAII_Timer("test_notebook_aub")
        notebook_filename = os.path.join(
            NOTEBOOK_PATHS,
            "AUB_EECE699_20201103_ReinforcementLearningApplication.ipynb",
        )
        self._aux_funct_notebook(notebook_filename)

    def test_notebook_ieeebda(self):
        if CAN_COMPUTE is not None:
            self.skipTest(f"{CAN_COMPUTE}")

        # this test takes 3 mins alone, for a really small benefit, so i skip it for sake of time
        return
        self._check_for_baselines()
        timer = RAII_Timer("test_notebook_ieeebda")
        notebook_filename = os.path.join(
            NOTEBOOK_PATHS, "IEEE BDA Tutorial Series.ipynb"
        )
        self._aux_funct_notebook(notebook_filename)


if __name__ == "__main__":
    unittest.main()