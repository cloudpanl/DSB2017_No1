# coding=utf-8
from __future__ import print_function

from pyspark.ml.feature import OneHotEncoderEstimator

from suanpan.arguments import Bool, ListOfString, String
from suanpan.spark import SparkComponent as sc
from suanpan.spark.arguments import HiveTable


@sc.input(HiveTable(key="inputData", table="inputTable", partition="inputPartition"))
@sc.output(
    HiveTable(key="outputData", table="outputTable", partition="outputPartition")
)
@sc.column(ListOfString(key="inputCols", required=True))
@sc.column(ListOfString(key="outputCols", required=True))
@sc.param(String(key="handleInvalid", default="error"))
@sc.param(Bool(key="dropLast", default=True))
def SPOneHotEncoderEstimator(context):
    args = context.args

    transorfomer = OneHotEncoderEstimator(
        inputCol=args.inputCol, outputCol=args.outputCol, **args.params
    )
    output = transorfomer.fit(args.inputData).transform(args.inputData)
    return output


if __name__ == "__main__":
    SPOneHotEncoderEstimator()  # pylint: disable=no-value-for-parameter
