# coding=utf-8
from __future__ import print_function

from pyspark.ml.feature import OneHotEncoder

from suanpan.arguments import Bool, String
from suanpan.spark import SparkComponent as sc
from suanpan.spark.arguments import HiveTable


@sc.input(HiveTable(key="inputData", table="inputTable", partition="inputPartition"))
@sc.output(
    HiveTable(key="outputData", table="outputTable", partition="outputPartition")
)
@sc.column(String(key="inputCol", required=True))
@sc.column(String(key="outputCol", required=True))
@sc.param(Bool(key="dropLast", default=True))
def SPOneHotEncoder(context):
    args = context.args

    transorfomer = OneHotEncoder(
        inputCol=args.inputCol, outputCol=args.outputCol, **args.params
    )
    output = transorfomer.transform(args.inputData)
    return output


if __name__ == "__main__":
    SPOneHotEncoder()  # pylint: disable=no-value-for-parameter
