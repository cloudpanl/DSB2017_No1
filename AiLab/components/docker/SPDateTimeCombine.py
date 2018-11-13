# coding=utf-8
from __future__ import print_function

import pandas as pd

from suanpan.arguments import ListOfString, String
from suanpan.docker import DockerComponent as dc
from suanpan.docker.arguments import HiveTable

DEFAULT_TIME_FEATURES = {
    "年": "year",
    "月": "month",
    "日": "day",
    "时": "hour",
    "分": "minute",
    "秒": "second",
}


@dc.input(HiveTable(key="inputData", table="inputTable", partition="inputPartition"))
@dc.output(
    HiveTable(key="outputData", table="outputTable", partition="outputPartition")
)
@dc.column(ListOfString(key="selectedColumns", default=DEFAULT_TIME_FEATURES.values()))
@dc.column(String(key="dataTimeColumn", default="datetime"))
@dc.column(String(key="dataTimeStringColumn", default="datetime_string"))
def SPDateTimeCombine(context):
    args = context.args
    data = args.inputData

    timeFeatures = {c: DEFAULT_TIME_FEATURES.get(c, c) for c in args.selectedColumns}
    data = data.rename(index=str, columns=timeFeatures)
    data[args.dataTimeColumn] = pd.to_datetime(data[list(timeFeatures.values())])
    data[args.dataTimeStringColumn] = data[args.dataTimeColumn].apply(
        lambda d: d.strftime("%Y-%m-%d %H:%M:%S")
    )
    return data


if __name__ == "__main__":
    SPDateTimeCombine()  # pylint: disable=no-value-for-parameter
