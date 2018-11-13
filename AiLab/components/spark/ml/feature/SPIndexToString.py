# coding=utf-8
from __future__ import print_function

from pyspark.ml.feature import IndexToString

from suanpan.arguments import ListOfString, String
from suanpan.spark import SparkComponent as sc
from suanpan.spark.arguments import HiveTable


@sc.input(HiveTable(key="inputData", table="inputTable", partition="inputPartition"))
@sc.output(
    HiveTable(key="outputData", table="outputTable", partition="outputPartition")
)
@sc.column(String(key="inputCol", required=True))
@sc.column(String(key="outputCol", required=True))
@sc.param(ListOfString(key="labels", required=True))
def SPIndexToString(context):
    args = context.args

    transorfomer = IndexToString(
        inputCol=args.inputCol, outputCol=args.outputCol, **args.params
    )
    output = transorfomer.transform(args.inputData)
    return output


if __name__ == "__main__":
    SPIndexToString()  # pylint: disable=no-value-for-parameter
