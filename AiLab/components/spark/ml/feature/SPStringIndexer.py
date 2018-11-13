# coding=utf-8
from __future__ import print_function

from pyspark.ml.feature import StringIndexer

from suanpan.arguments import String
from suanpan.spark import SparkComponent as sc
from suanpan.spark.arguments import HiveTable


@sc.input(HiveTable(key="inputData", table="inputTable", partition="inputPartition"))
@sc.output(
    HiveTable(key="outputData", table="outputTable", partition="outputPartition")
)
@sc.column(String(key="inputCol", required=True))
@sc.column(String(key="outputCol", required=True))
@sc.param(String(key="handleInvalid", default="error"))
@sc.param(String(key="stringOrderType", default="frequencyDesc"))
def SPStringIndexer(context):
    args = context.args

    transorfomer = StringIndexer(
        inputCol=args.inputCol, outputCol=args.outputCol, **args.params
    )
    output = transorfomer.fit(args.inputData).transform(args.inputData)
    return output


if __name__ == "__main__":
    SPStringIndexer()  # pylint: disable=no-value-for-parameter
