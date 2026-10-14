# Copyright (c) 2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import hashlib
import os
import shutil
import tarfile
import tempfile
import unittest
import warnings
from unittest import mock

import grid2op.Download.DownloadDataset as dl
import grid2op.MakeEnv.Make as make_mod
from grid2op.Exceptions import Grid2OpException

ENV_NAME = "fake_env"
URL = "https://example.invalid/datasets/fake_env.tar.bz2"


class TestDownloadChecksum(unittest.TestCase):
    """The archive of a dataset is verified (SHA-256) before anything is extracted."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.path_data = os.path.join(self.tmp, "data")
        src = os.path.join(self.tmp, "src", ENV_NAME)
        os.makedirs(src)
        with open(os.path.join(src, "config.py"), "w") as f:
            f.write("config = {}\n")
        self.archive = os.path.join(self.tmp, "fake_env.tar.bz2")
        with tarfile.open(self.archive, "w:bz2") as tar:
            tar.add(src, arcname=ENV_NAME)
        with open(self.archive, "rb") as f:
            self.good_sha = hashlib.sha256(f.read()).hexdigest()

        def fake_download(url, output_path):
            shutil.copyfile(self.archive, output_path)

        # no network: the "download" copies the local archive, and no remote update is attempted
        patches = [
            mock.patch.object(dl, "download_url", side_effect=fake_download),
            mock.patch("grid2op.MakeEnv.UpdateEnv._update_files"),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _download(self, **kwargs):
        dl._aux_download(URL, ENV_NAME, self.path_data, **kwargs)

    def test_sha256_file(self):
        assert dl._sha256_file(self.archive) == self.good_sha
        # a block size smaller than the file must give the same digest
        assert dl._sha256_file(self.archive, blocksize=7) == self.good_sha

    def test_good_checksum_extracts(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._download(sha256=self.good_sha)
        # a verified download must not complain about a missing checksum
        assert not [w for w in caught if "SHA-256" in str(w.message)]
        assert os.path.exists(os.path.join(self.path_data, ENV_NAME, "config.py"))
        assert not os.path.exists(os.path.join(self.path_data, "fake_env.tar.bz2"))

    def test_good_checksum_is_case_insensitive(self):
        self._download(sha256=self.good_sha.upper())
        assert os.path.exists(os.path.join(self.path_data, ENV_NAME, "config.py"))

    def test_bad_checksum_refuses_to_extract(self):
        bad = "0" * 64
        with self.assertRaises(Grid2OpException) as cm:
            self._download(sha256=bad)
        assert "SHA-256" in str(cm.exception)
        assert self.good_sha in str(cm.exception)
        # nothing extracted, and the corrupted archive is removed so a retry downloads it again
        assert not os.path.exists(os.path.join(self.path_data, ENV_NAME))
        assert not os.path.exists(os.path.join(self.path_data, "fake_env.tar.bz2"))

    def test_tampered_archive_refuses_to_extract(self):
        with open(self.archive, "ab") as f:
            f.write(b"tampered")
        with self.assertRaises(Grid2OpException):
            self._download(sha256=self.good_sha)
        assert not os.path.exists(os.path.join(self.path_data, ENV_NAME))

    def test_archive_name_cannot_escape_data_dir(self):
        with self.assertRaises(Grid2OpException):
            dl._aux_download(URL, ENV_NAME, self.path_data, ds_name_dl="../escaped", sha256=self.good_sha)
        assert not os.path.exists(os.path.join(self.tmp, "escaped.tar.bz2"))

    def test_missing_checksum_warns_but_extracts(self):
        with self.assertWarnsRegex(UserWarning, "SHA-256"):
            self._download()
        assert os.path.exists(os.path.join(self.path_data, ENV_NAME, "config.py"))


class TestFetchEnvironmentsChecksum(unittest.TestCase):
    """``datasets.json`` entries can carry a ``sha256`` that ``make`` forwards to the download."""

    def _fetch(self, entry):
        with mock.patch.object(
            make_mod, "_list_available_remote_env_aux", return_value={ENV_NAME: entry}
        ):
            return make_mod._fecth_environments(ENV_NAME)

    def test_sha256_is_returned(self):
        entry = {
            "base_url": "https://example.invalid/datasets/",
            "filename": "fake_env.tar.bz2",
            "sha256": "ab" * 32,
        }
        url, ds_name_dl, sha256 = self._fetch(entry)
        assert url == URL
        assert ds_name_dl == ENV_NAME
        assert sha256 == "ab" * 32

    def test_sha256_is_optional(self):
        entry = {
            "base_url": "https://example.invalid/datasets/",
            "filename": "fake_env.tar.bz2",
        }
        _, _, sha256 = self._fetch(entry)
        assert sha256 is None


if __name__ == "__main__":
    unittest.main()
