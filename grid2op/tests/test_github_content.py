# Copyright (c) 2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import unittest
from unittest import mock

import grid2op.MakeEnv.Make as make_mod
from grid2op.Exceptions import Grid2OpException

API_URL = "https://api.github.com/repos/Grid2Op/grid2op-datasets/contents/datasets.json"
RAW_URL = "https://raw.githubusercontent.com/Grid2Op/grid2op-datasets/HEAD/datasets.json"


class _FakeResponse:
    def __init__(self, text='{"a": 1}'):
        self.text = text

    def json(self):
        import json

        return json.loads(self.text)


class TestGithubRawUrl(unittest.TestCase):
    def test_api_url_is_converted(self):
        assert make_mod._github_raw_url(API_URL) == RAW_URL

    def test_nested_path_and_other_owner(self):
        url = "https://api.github.com/repos/someone/grid2op-datasets/contents/updates/a_config.py"
        assert (
            make_mod._github_raw_url(url)
            == "https://raw.githubusercontent.com/someone/grid2op-datasets/HEAD/updates/a_config.py"
        )

    def test_other_urls_are_unchanged(self):
        for url in [RAW_URL, "https://example.invalid/datasets.json"]:
            assert make_mod._github_raw_url(url) == url


class TestRetrieveGithubContent(unittest.TestCase):
    def setUp(self):
        # no real waiting and no state leaking between tests
        for p in [
            mock.patch.object(make_mod.time, "sleep"),
            mock.patch.object(make_mod, "_last_github_request", None),
        ]:
            p.start()
            self.addCleanup(p.stop)

    def test_single_request_to_raw_url(self):
        with mock.patch.object(
            make_mod, "_send_request_retry", return_value=_FakeResponse()
        ) as send:
            res = make_mod._retrieve_github_content(API_URL)
        assert res == {"a": 1}
        # the rate limited api is not used anymore: one request, straight to the raw content
        send.assert_called_once_with(RAW_URL)

    def test_text_content(self):
        with mock.patch.object(
            make_mod, "_send_request_retry", return_value=_FakeResponse("config = {}\n")
        ):
            res = make_mod._retrieve_github_content(API_URL, is_json=False)
        assert res == "config = {}\n"

    def test_invalid_json(self):
        with mock.patch.object(
            make_mod, "_send_request_retry", return_value=_FakeResponse("not json")
        ):
            with self.assertRaises(Grid2OpException):
                make_mod._retrieve_github_content(API_URL)


class TestGithubDelay(unittest.TestCase):
    """At least one second between two requests to github, but no useless wait."""

    def setUp(self):
        self.now = 1000.0
        self.slept = []

        def fake_sleep(duration):
            self.slept.append(duration)
            self.now += duration

        for p in [
            mock.patch.object(make_mod, "_last_github_request", None),
            mock.patch.object(make_mod.time, "monotonic", side_effect=lambda: self.now),
            mock.patch.object(make_mod.time, "sleep", side_effect=fake_sleep),
        ]:
            p.start()
            self.addCleanup(p.stop)

    def test_no_wait_for_first_request(self):
        make_mod._wait_before_github_request()
        assert self.slept == []

    def test_wait_between_consecutive_requests(self):
        make_mod._wait_before_github_request()
        self.now += 0.25  # the first request took 0.25s
        make_mod._wait_before_github_request()
        assert len(self.slept) == 1
        assert abs(self.slept[0] - 0.75) < 1e-9

    def test_no_wait_if_enough_time_elapsed(self):
        make_mod._wait_before_github_request()
        self.now += 5.0
        make_mod._wait_before_github_request()
        assert self.slept == []


if __name__ == "__main__":
    unittest.main()
