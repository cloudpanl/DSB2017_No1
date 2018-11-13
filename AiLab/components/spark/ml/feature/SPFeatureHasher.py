# coding=utf-8
from __future__ import print_function

from pyspark.ml.feature import FeatureHasher

from suanpan.arguments import Int, ListOfString, String
from suanpan.spark import SparkComponent as sc
from suanpan.spark.arguments import HiveTable


@sc.input(HiveTable(key="inputData", table="inputTable", partition="inputPartition"))
@sc.output(
    HiveTable(key="outputData", table="outputTable", partition="outputPartition")
)
@sc.column(ListOfString(key="inputCols", required=True))
@sc.column(String(key="outputCol", required=True))
@sc.column(ListOfString(key="categoricalCols"))
@sc.param(Int(key="numFeatures", default=262144))
def SPFeatureHasher(context):
    args = context.args

    transorfomer = FeatureHasher(
        inputCol=args.inputCols,
        outputCol=args.outputCol,
        categoricalCols=args.categoricalCols,
        **args.params
    )
    output = transorfomer.transform(args.inputData)
    return output


if __name__ == "__main__":
    SPFeatureHasher()  # pylint: disable=no-value-for-parameter
