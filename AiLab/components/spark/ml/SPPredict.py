# coding=utf-8
from __future__ import print_function

from suanpan.spark import SparkComponent as sc
from suanpan.spark.arguments import HiveTable, SparkModel


@sc.input(SparkModel(key="inputModel"))
@sc.input(HiveTable(key="inputData", table="inputTable", partition="inputPartition"))
@sc.output(
    HiveTable(key="predictionData", table="outputTable", partition="outputPartition")
)
def SPPredictor(context):
    args = context.args

    predictionData = args.inputModel.transform(args.inputData)
    return predictionData


if __name__ == "__main__":
    SPPredictor()  # pylint: disable=no-value-for-parameter
