# coding=utf-8
from __future__ import print_function

from suanpan.arguments import String
from suanpan.docker.io import storage
from suanpan.tools import ToolComponent as tc


@tc.param(String(key="action", required=True))
@tc.param(String(key="local"))
@tc.param(String(key="remote", required=True))
def SPStorageTools(context):
    args = context.args

    if args.action == "download":
        storage.download(args.remote, args.local)
    elif args.action == "upload":
        storage.upload(args.remote, args.local)
    elif args.action == "remove":
        storage.remove(args.remote)
    else:
        raise Exception("Unsupport action: {}".format(args.action))


if __name__ == "__main__":
    SPStorageTools()  # pylint: disable=no-value-for-parameter
