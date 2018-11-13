# coding=utf-8
from __future__ import print_function

from pyspark.ml.feature import StopWordsRemover

from suanpan.arguments import Bool, ListOfString, String
from suanpan.spark import SparkComponent as sc
from suanpan.spark.arguments import HiveTable


@sc.input(HiveTable(key="inputData", table="inputTable", partition="inputPartition"))
@sc.output(
    HiveTable(key="outputData", table="outputTable", partition="outputPartition")
)
@sc.column(String(key="inputCol", required=True))
@sc.column(String(key="outputCol", required=True))
@sc.param(ListOfString(key="stopWords", required=True))
@sc.param(Bool(key="caseSensitive", default=False))
def SPStopWordsRemover(context):
    args = context.args

    transorfomer = StopWordsRemover(
        inputCol=args.inputCol, outputCol=args.outputCol, **args.params
    )
    output = transorfomer.transform(args.inputData)
    return output


if __name__ == "__main__":
    SPStopWordsRemover()  # pylint: disable=no-value-for-parameter
