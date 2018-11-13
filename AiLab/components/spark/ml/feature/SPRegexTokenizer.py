# coding=utf-8
from __future__ import print_function

from pyspark.ml.feature import RegexTokenizer

from suanpan.arguments import Bool, Int, String
from suanpan.spark import SparkComponent as sc
from suanpan.spark.arguments import HiveTable


@sc.input(HiveTable(key="inputData", table="inputTable", partition="inputPartition"))
@sc.output(
    HiveTable(key="outputData", table="outputTable", partition="outputPartition")
)
@sc.column(String(key="inputCol", required=True))
@sc.column(String(key="outputCol", required=True))
@sc.param(Int(key="minTokenLength", default=1))
@sc.param(Bool(key="gaps", default=True))
@sc.param(String(key="pattern", default=r"\s+"))
@sc.param(Bool(key="toLowercase", default=True))
def SPRegexTokenizer(context):
    args = context.args

    transorfomer = RegexTokenizer(
        inputCol=args.inputCol, outputCol=args.outputCol, **args.params
    )
    output = transorfomer.transform(args.inputData)
    return output


if __name__ == "__main__":
    SPRegexTokenizer()  # pylint: disable=no-value-for-parameter
