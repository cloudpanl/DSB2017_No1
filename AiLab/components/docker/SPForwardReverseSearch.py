# coding=utf-8
from __future__ import print_function

import numpy as np
import pandas as pd
from suanpan.arguments import Int, ListOfString, String
from suanpan.docker import DockerComponent as dc
from suanpan.docker.arguments import HiveTable
from suanpan.log import logger


def setStage(data, stage, stageColumn="stage"):
    if not data.empty:
        data.loc[:, stageColumn] = stage
    return data


def setGroupStages(args, period, group, stageColumn="stage", errorStage=-1):
    reversedGroup = group.iloc[::-1]  # pylint: disable=unused-variable
    forwardResult = runEval(args.forwardCondition, group, args.selectedColumns)
    reverseResult = runEval(args.reverseCondition, reversedGroup, args.selectedColumns)
    forwardIndex = forwardResult.argmax()
    reverseIndex = len(group) - reverseResult.argmax()

    if forwardResult.any() and reverseResult.any() and forwardIndex <= reverseIndex:
        stages = (
            group[:forwardIndex],
            group[forwardIndex:reverseIndex],
            group[reverseIndex:],
        )
        stages = [
            setStage(stage, index, stageColumn=stageColumn)
            for index, stage in enumerate(stages)
        ]
        logger.info(
            "Period {period}: [:{forwardIndex}, {forwardIndex}:{reverseIndex}, {reverseIndex}:].".format(
                period=period, forwardIndex=forwardIndex, reverseIndex=reverseIndex
            )
        )
        return pd.concat(stages, sort=False)
    else:
        logger.info("Period {period}: Mark as error.")
        return setStage(group, errorStage, stageColumn=stageColumn)


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
@dc.column(String(key="stageColumn", default="stage"))
@dc.param(String(key="forwardCondition", required=True))
@dc.param(String(key="reverseCondition", required=True))
@dc.param(Int(key="errorStage", default=-1))
def SPForwardReverseSearch(context):
    args = context.args
    data = args.inputData

    groups = [
        setGroupStages(
            args,
            period,
            group,
            stageColumn=args.stageColumn,
            errorStage=args.errorStage,
        )
        for period, group in data.groupby(args.periodColumn)
    ]
    output = pd.concat(groups, sort=False)
    return output


if __name__ == "__main__":
    SPForwardReverseSearch()  # pylint: disable=no-value-for-parameter
