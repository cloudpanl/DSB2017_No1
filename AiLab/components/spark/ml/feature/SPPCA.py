# coding=utf-8
from __future__ import print_function

from pyspark.ml.feature import PCA

from suanpan.arguments import Int, String
from suanpan.spark import SparkComponent as sc
from suanpan.spark.arguments import HiveTable


@sc.input(HiveTable(key="inputData", table="inputTable", partition="inputPartition"))
@sc.output(
    HiveTable(key="outputData", table="outputTable", partition="outputPartition")
)
@sc.column(String(key="inputCol", required=True))
@sc.column(String(key="outputCol", required=True))
@sc.param(Int(key="k", required=True))
def SPPCA(context):
    args = context.args

    transorfomer = PCA(inputCol=args.inputCol, outputCol=args.outputCol, **args.params)
    output = transorfomer.fit(args.inputData).transform(args.inputData)
    return output


if __name__ == "__main__":
    SPPCA()  # pylint: disable=no-value-for-parameter
