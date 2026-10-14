#!/usr/bin/env python3
# Copyright (c) 2019-2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

"""
Split a branch's diff into package code / tests / data / docs and print it as a markdown
table, to put at the top of a pull request body.

A PR here routinely runs to one or two thousand lines, which is daunting to open and says
nothing about where the work actually is. The table says how much of it is code a reviewer
has to reason about and how much is tests, bundled data, docs and changelog::

    python3 utils/pr_diff_stats.py                 # against the highest origin/dev_X.Y.Z (the default)
    python3 utils/pr_diff_stats.py origin/master   # against another base branch

Counts the branch's own changes only (`git diff base...HEAD`), so commits that landed on
the base branch meanwhile are not attributed to the PR.
"""

import re
import subprocess
import sys

#: (bucket key, label), in the order the table prints them
BUCKETS = [
    ("python", "Python (grid2op package)"),
    ("tests", "Tests (grid2op/tests)"),
    ("data", "Data (grid2op/data, grid2op/data_test)"),
    ("docs", "Docs + changelog + notebooks"),
    ("other", "CI / build / other"),
]


def bucket_of(path):
    """Which row of the table a changed file belongs to.

    Tests and data come first on purpose: a ``.py`` file under ``grid2op/tests/`` is a
    test, and a ``config.py`` under ``grid2op/data/`` is data, not package code.
    """
    if path.startswith("grid2op/tests/"):
        return "tests"
    if path.startswith(("grid2op/data/", "grid2op/data_test/")):
        return "data"
    if path.startswith(("docs/", "getting_started/", "examples/")) or path.endswith((".rst", ".md", ".ipynb")):
        return "docs"
    if path.startswith("grid2op/") and path.endswith(".py"):
        return "python"
    return "other"


def default_base():
    """The highest ``origin/dev_X.Y.Z`` branch: pull requests here target the active dev branch."""
    out = subprocess.run(["git", "branch", "-r", "--list", "origin/dev_*"],
                         capture_output=True, text=True, check=True).stdout
    candidates = []
    for name in out.split():
        match = re.fullmatch(r"origin/dev_(\d+)\.(\d+)\.(\d+)", name)
        if match:
            candidates.append((tuple(int(x) for x in match.groups()), name))
    if not candidates:
        raise RuntimeError("no origin/dev_X.Y.Z branch found, pass the base branch explicitly")
    return max(candidates)[1]


def diff_stats(base):
    """{bucket: [added, removed]} for `git diff base...HEAD`."""
    merge_base = subprocess.run(["git", "merge-base", base, "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
    numstat = subprocess.run(["git", "diff", "--numstat", merge_base + "...HEAD"],
                             capture_output=True, text=True, check=True).stdout
    totals = {}
    for line in numstat.strip().splitlines():
        added, removed, path = line.split("\t")
        if added == "-":
            continue  # binary file: git reports no line counts
        entry = totals.setdefault(bucket_of(path), [0, 0])
        entry[0] += int(added)
        entry[1] += int(removed)
    return totals


def as_markdown(totals):
    lines = ["| | added | removed |", "|---|---|---|"]
    for key, label in BUCKETS:
        if key in totals:  # an empty bucket gets no row
            lines.append(f"| **{label}** | +{totals[key][0]} | -{totals[key][1]} |")
    added = sum(v[0] for v in totals.values())
    removed = sum(v[1] for v in totals.values())
    lines.append(f"| total | +{added} | -{removed} |")
    return "\n".join(lines)


def main():
    try:
        base = sys.argv[1] if len(sys.argv) > 1 else default_base()
        table = as_markdown(diff_stats(base))
        print(f"diff against {base}", file=sys.stderr)  # stdout is only the table, ready to paste
        print(table)
    except subprocess.CalledProcessError as exc:
        print(f"git failed: {exc.stderr.strip() or exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
