# coding=utf-8
from __future__ import print_function

import pandas as pd

from suanpan import utils
from suanpan.arguments import String
from suanpan.docker.datawarehouse import dw
from suanpan.tools import ToolComponent as tc


@tc.param(String(key="action", required=True))
@tc.param(String(key="file"))
@tc.param(String(key="table"))
def SPDWTools(context):
    args = context.args

    if args.action == "upload":
        data = utils.loadFromCsv(args.file)
        dw.writeTable(args.table, data)
    elif args.action == "download":
        data = dw.readTable(args.table)
        data.to_csv(args.file)
    else:
        raise Exception("Unsupport action: {}".format(args.action))


if __name__ == "__main__":
    SPDWTools()  # pylint: disable=no-value-for-parameter
