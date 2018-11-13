# coding=utf-8
from __future__ import print_function

from pyspark.ml.feature import Bucketizer

from suanpan.arguments import ListOfFloat, String
from suanpan.spark import SparkComponent as sc
from suanpan.spark.arguments import HiveTable


@sc.input(HiveTable(key="inputData", table="inputTable", partition="inputPartition"))
@sc.output(
    HiveTable(key="outputData", table="outputTable", partition="outputPartition")
)
@sc.column(String(key="inputCol", required=True))
@sc.column(String(key="outputCol", required=True))
@sc.param(ListOfFloat(key="splits", required=True))
@sc.param(String(key="handleInvalid", default="error"))
def SPBucketizer(context):
    args = context.args

    transorfomer = Bucketizer(
        inputCol=args.inputCol, outputCol=args.outputCol, **args.params
    )
    output = transorfomer.transform(args.inputData)
    return output


if __name__ == "__main__":
    SPBucketizer()  # pylint: disable=no-value-for-parameter
