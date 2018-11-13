# coding=utf-8
from __future__ import print_function

import itertools

import numpy as np
import pandas as pd
from suanpan.arguments import ListOfString, String
from suanpan.docker import DockerComponent as dc
from suanpan.docker.arguments import HiveTable
from suanpan.log import logger


def setGroupPeriod(data, group, period, periodColumn="period"):
    logger.info("Find period {}: No.{} - No.{}".format(period, group[0], group[-1]))
    data.loc[group, periodColumn] = period
    return data.loc[group]


def runEval(evalString, data, columns, **kwargs):
    data = data[columns]
    kwargs.update(
        data=data,
        columns=columns,
        all=np.logical_and.reduce,  # pylint: disable=no-member
        any=np.logical_or.reduce,  # pylint: disable=no-member
        mean=data.mean(),
        std=data.std(),
        var=data.var(),
        sum=data.sum(),
    )
    return eval(evalString, kwargs)


@dc.input(HiveTable(key="inputData", table="inputTable", partition="inputPartition"))
@dc.output(
    HiveTable(key="outputData", table="outputTable", partition="outputPartition")
)
@dc.column(ListOfString(key="selectedColumns", required=True))
@dc.column(String(key="periodColumn", default="period"))
@dc.param(String(key="condition", required=True))
def SPWaveExtract(context):
    args = context.args
    data = args.inputData

    keeped = runEval(args.condition, data, args.selectedColumns)
    keepedIndexes = data[keeped].index
    seriesIndexesGroups = (
        [value[1] for value in values]
        for _, values in itertools.groupby(
            enumerate(keepedIndexes), lambda x: int(x[1]) - x[0]
        )
    )
    seriesGroups = [
        setGroupPeriod(data, group, period, args.periodColumn)
        for period, group in enumerate(seriesIndexesGroups)
    ]
    output = pd.concat(seriesGroups)
    return output


if __name__ == "__main__":
    SPWaveExtract()  # pylint: disable=no-value-for-parameter
