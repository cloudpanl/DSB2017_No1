# coding=utf-8
from __future__ import print_function

import pandas as pd

from suanpan.arguments import String
from suanpan.docker import DockerComponent as dc
from suanpan.docker.datawarehouse import dw


@dc.input(String(key="inputTable", required=True))
@dc.output(String(key="outputTable", required=True))
@dc.param(String(key="sql", required=True))
def SPHiveSql(context):
    args = context.args

    dw.execute(args.sql)


if __name__ == "__main__":
    SPHiveSql()  # pylint: disable=no-value-for-parameter
