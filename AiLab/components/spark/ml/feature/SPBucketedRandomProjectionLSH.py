# coding=utf-8
from __future__ import print_function

from pyspark.ml.feature import BucketedRandomProjectionLSH

from suanpan.arguments import Float, Int, String
from suanpan.spark import SparkComponent as sc
from suanpan.spark.arguments import HiveTable


@sc.input(HiveTable(key="inputData", table="inputTable", partition="inputPartition"))
@sc.output(
    HiveTable(key="outputData", table="outputTable", partition="outputPartition")
)
@sc.column(String(key="inputCol", required=True))
@sc.column(String(key="outputCol", required=True))
@sc.param(Int(key="seed"))
@sc.param(Int(key="numHashTables", default=1))
@sc.param(Float(key="bucketLength", required=True))
def SPBucketedRandomProjectionLSH(context):
    args = context.args

    transorfomer = BucketedRandomProjectionLSH(
        inputCol=args.inputCol, outputCol=args.outputCol, **args.params
    )
    output = transorfomer.fit(args.inputData).transform(args.inputData)
    return output


if __name__ == "__main__":
    SPBucketedRandomProjectionLSH()  # pylint: disable=no-value-for-parameter
