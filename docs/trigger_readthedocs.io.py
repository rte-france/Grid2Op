# This files allows to automatically update the documentation of grid2op on the
# readthedocs.io website. It should not be used for other purpose.

try:
    import requests as rq
except:
    raise RuntimeError("Impossible to find library urllib. Please install it.")

import json
import os

if not os.path.exists(".readthedocstoken.json"):
    raise RuntimeError("Impossible to find credential for buildthedocs. Stopping there. Make sur to put them on \".readthedocstoken.json\"")

with open(".readthedocstoken.json", "r") as f:
    dict_credentials = json.load(f)

token = dict_credentials["token"]
# curl \
#   -X POST \
#   -H "Authorization: Token <token>" https://readthedocs.org/api/v3/projects/pip/versions/latest/builds/

url = "https://readthedocs.org/api/v3/projects/grid2op/versions/{version_slug}/builds/"
version = "latest"
hdr = {"Authorization": "Token {}".format(token)}
req = rq.post(url.format(version_slug=version), headers=hdr)
print(req.text)