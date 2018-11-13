# coding=utf-8
from __future__ import print_function

from pyspark.ml.feature import HashingTF

from suanpan.arguments import Bool, Int, String
from suanpan.spark import SparkComponent as sc
from suanpan.spark.arguments import HiveTable


@sc.input(HiveTable(key="inputData", table="inputTable", partition="inputPartition"))
@sc.output(
    HiveTable(key="outputData", table="outputTable", partition="outputPartition")
)
@sc.column(String(key="inputCol", required=True))
@sc.column(String(key="outputCol", required=True))
@sc.param(Int(key="numFeatures", default=262144))
@sc.param(Bool(key="binary", default=False))
def SPHashingTF(context):
    args = context.args

    transorfomer = HashingTF(
        inputCol=args.inputCol, outputCol=args.outputCol, **args.params
    )
    output = transorfomer.transform(args.inputData)
    return output


if __name__ == "__main__":
    SPHashingTF()  # pylint: disable=no-value-for-parameter
