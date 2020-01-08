# This files allows to automatically update the documentation of grid2op on the
# readthedocs.io website. It should not be used for other purpose.

import argparse
import json
import os
import re
import time

try:
    import requests as rq
except:
    raise RuntimeError("Impossible to find library urllib. Please install it.")

import pdb

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Update the version of grid2op in the python files.')
    parser.add_argument('--version', default=None,
                        help='The new version to update.')
    parser.add_argument('--path', default=os.path.abspath("."),
                        help='The path of the root directory of Grid2op (default {}'.format(os.path.abspath(".")))

    args = parser.parse_args()
    path = args.path
    version = args.version

    if args.version is None:
        raise RuntimeError("script \"update_version\" should be called with a version number.")

    try:
        maj_, min_, minmin_ = version.split(".")
    except:
        raise RuntimeError("script \"update_version\": version should be formated as XX.YY.ZZ (eg 0.3.1). Please modify \"--version\" argument")

    if re.match('^[0-9]+\.[0-9]+\.[0-9]+$', version) is None:
        raise RuntimeError("script \"update_version\": version should be formated as XX.YY.ZZ (eg 0.3.1) and not {}. Please modify \"--version\" argument".format(version))

    if not os.path.exists(".readthedocstoken.json"):
        raise RuntimeError("Impossible to find credential for buildthedocs. Stopping there. Make sur to put them on \".readthedocstoken.json\"")

    with open(".readthedocstoken.json", "r") as f:
        dict_credentials = json.load(f)

    token = dict_credentials["token"]
    hdr = {"Authorization": "Token {}".format(token)}

    # curl \
    #   -X POST \
    #   -H "Authorization: Token <token>" https://readthedocs.org/api/v3/projects/pip/versions/latest/builds/

    # list existing version on read the doc:
    url_existing_version = "https://readthedocs.org/api/v3/projects/grid2op/versions/"
    req = rq.get(url_existing_version, headers=hdr)
    resp = req.json()
    li_existing_version = set()
    for el in resp["results"]:
        li_existing_version.add(el['slug'])

    # update new versions
    url_version = "https://readthedocs.org/api/v3/projects/grid2op/versions/{version_slug}/builds/"

    for vers_ in ["v{}".format(version), "stable", "latest"]:
        if vers_ in li_existing_version:
            req = rq.post(url_version.format(version_slug=vers_), headers=hdr)
            print("Version {} properly updated".format(vers_))
            time.sleep(5)
        else:
            raise RuntimeError("Version \"{}\" is not part of the read the doc version,"
                               "please create it before updating it.".format(vers_))
