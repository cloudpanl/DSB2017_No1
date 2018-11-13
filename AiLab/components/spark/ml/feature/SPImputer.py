# coding=utf-8
from __future__ import print_function

from pyspark.ml.feature import Imputer

from suanpan.arguments import Float, ListOfString, String
from suanpan.spark import SparkComponent as sc
from suanpan.spark.arguments import HiveTable


@sc.input(HiveTable(key="inputData", table="inputTable", partition="inputPartition"))
@sc.output(
    HiveTable(key="outputData", table="outputTable", partition="outputPartition")
)
@sc.column(ListOfString(key="inputCols", required=True))
@sc.column(ListOfString(key="outputCols", required=True))
@sc.param(String(key="strategy", default="mean"))
@sc.param(Float(key="missingValue", default=float("nan")))
def SPImputer(context):
    args = context.args

    transorfomer = Imputer(
        inputCols=args.inputCols, outputCols=args.outputCols, **args.params
    )
    output = transorfomer.fit(args.inputData).transform(args.inputData)
    return output


if __name__ == "__main__":
    SPImputer()  # pylint: disable=no-value-for-parameter
