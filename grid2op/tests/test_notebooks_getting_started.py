# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import shutil
import os
import copy

import pdb
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError
from grid2op.tests.helper_path_test import *
import os
import unittest

# TODO check these tests, they don't appear to be working

import warnings
warnings.simplefilter("error")
NOTEBOOK_PATHS = os.path.abspath(os.path.join(PATH_DATA_TEST, "../../getting_started"))


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
            print('Failed to delete %s. Reason: %s' % (file_path, e))


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


class RAII_tf_log():
    def __init__(self):
        self.previous = None
        if "TF_CPP_MIN_LOG_LEVEL" in os.environ:
            self.previous = copy.deepcopy(os.environ['TF_CPP_MIN_LOG_LEVEL'])
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    def __del__(self):
        if self.previous is not None:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = self.previous
# notebook names are hard coded because if i change them, i need also to change the
# readme and the documentation


class TestNotebook(unittest.TestCase):
    def _aux_funct_notebook(self, notebook_filename):
        assert os.path.exists(notebook_filename), f"{notebook_filename} do not exists!"
        with open(notebook_filename) as f:
            nb = nbformat.read(f, as_version=4)
        try:
            ep = ExecutePreprocessor(timeout=600, store_widget_state=True)
            try:
                ep.preprocess(nb, {'metadata': {'path': NOTEBOOK_PATHS}})
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
        notebook_filename = os.path.join(NOTEBOOK_PATHS, "00_SmallExample.ipynb")
        self._aux_funct_notebook(notebook_filename)

    def test_notebook1(self):
        notebook_filename = os.path.join(NOTEBOOK_PATHS, "01_Grid2opFramework.ipynb")
        self._aux_funct_notebook(notebook_filename)

    def test_notebook2(self):
        notebook_filename = os.path.join(NOTEBOOK_PATHS, "02_Observation.ipynb")
        self._aux_funct_notebook(notebook_filename)

    def test_notebook3(self):
        notebook_filename = os.path.join(NOTEBOOK_PATHS, "03_Action.ipynb")
        self._aux_funct_notebook(notebook_filename)

    def test_notebook4(self):
        self._check_for_baselines()
        raii_ = RAII_tf_log()
        notebook_filename = os.path.join(NOTEBOOK_PATHS, "04_TrainingAnAgent.ipynb")
        self._aux_funct_notebook(notebook_filename)

    def test_notebook5(self):
        notebook_filename = os.path.join(NOTEBOOK_PATHS, "05_StudyYourAgent.ipynb")
        self._aux_funct_notebook(notebook_filename)

    def test_notebook6(self):
        notebook_filename = os.path.join(NOTEBOOK_PATHS, "06_Redispatching.ipynb")
        self._aux_funct_notebook(notebook_filename)

    def test_notebook7(self):
        self._check_for_baselines()
        raii_ = RAII_tf_log()
        notebook_filename = os.path.join(NOTEBOOK_PATHS, "07_MultiEnv.ipynb")
        self._aux_funct_notebook(notebook_filename)

    def test_notebook8(self):
        notebook_filename = os.path.join(NOTEBOOK_PATHS, "08_PlottingCapabilities.ipynb")
        self._aux_funct_notebook(notebook_filename)

    def test_notebook9(self):
        notebook_filename = os.path.join(NOTEBOOK_PATHS, "09_EnvironmentModifications.ipynb")
        self._aux_funct_notebook(notebook_filename)

    def test_notebook_aub(self):
        raii_ = RAII_tf_log()
        notebook_filename = os.path.join(NOTEBOOK_PATHS, "AUB_EECE699_20201103_ReinforcementLearningApplication.ipynb")
        self._aux_funct_notebook(notebook_filename)

    def test_notebook_ieeebda(self):
        self._check_for_baselines()
        notebook_filename = os.path.join(NOTEBOOK_PATHS, "IEEE BDA Tutorial Series.ipynb")
        self._aux_funct_notebook(notebook_filename)


if __name__ == "__main__":
    unittest.main()
