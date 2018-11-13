# coding=utf-8
from __future__ import print_function

from pyspark.ml.feature import CountVectorizer

from suanpan.arguments import Bool, Float, Int, String
from suanpan.spark import SparkComponent as sc
from suanpan.spark.arguments import HiveTable


@sc.input(HiveTable(key="inputData", table="inputTable", partition="inputPartition"))
@sc.output(
    HiveTable(key="outputData", table="outputTable", partition="outputPartition")
)
@sc.column(String(key="inputCol", required=True))
@sc.column(String(key="outputCol", required=True))
@sc.param(Float(key="minTF", default=1.0))
@sc.param(Float(key="minDF", default=1.0))
@sc.param(Int(key="vocabSize", default=262144))
@sc.param(Bool(key="binary", default=False))
def SPCountVectorizer(context):
    args = context.args

    transorfomer = CountVectorizer(
        inputCol=args.inputCol, outputCol=args.outputCol, **args.params
    )
    output = transorfomer.fit(args.inputData).transform(args.inputData)
    return output


if __name__ == "__main__":
    SPCountVectorizer()  # pylint: disable=no-value-for-parameter
