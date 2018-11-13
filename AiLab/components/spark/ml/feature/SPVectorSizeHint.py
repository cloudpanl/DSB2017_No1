# coding=utf-8
from __future__ import print_function

from pyspark.ml.feature import VectorSizeHint

from suanpan.arguments import Int, String
from suanpan.spark import SparkComponent as sc
from suanpan.spark.arguments import HiveTable


@sc.input(HiveTable(key="inputData", table="inputTable", partition="inputPartition"))
@sc.output(
    HiveTable(key="outputData", table="outputTable", partition="outputPartition")
)
@sc.column(String(key="inputCol", required=True))
@sc.param(Int(key="size", required=True))
@sc.param(String(key="handleInvalid", default="error"))
def SPVectorSizeHint(context):
    args = context.args

    transorfomer = VectorSizeHint(inputCol=args.inputCol, **args.params)
    output = transorfomer.transform(args.inputData)
    return output


if __name__ == "__main__":
    SPVectorSizeHint()  # pylint: disable=no-value-for-parameter
