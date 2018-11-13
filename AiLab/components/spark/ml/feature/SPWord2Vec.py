# coding=utf-8
from __future__ import print_function

from pyspark.ml.feature import Word2Vec

from suanpan.arguments import Float, Int, String
from suanpan.spark import SparkComponent as sc
from suanpan.spark.arguments import HiveTable


@sc.input(HiveTable(key="inputData", table="inputTable", partition="inputPartition"))
@sc.output(
    HiveTable(key="outputData", table="outputTable", partition="outputPartition")
)
@sc.column(String(key="inputCol", required=True))
@sc.column(String(key="outputCol", required=True))
@sc.param(Int(key="vectorSize", default=100))
@sc.param(Int(key="minCount", default=5))
@sc.param(Int(key="numPartitions", default=1))
@sc.param(Float(key="stepSize", default=0.025))
@sc.param(Int(key="maxIter", default=1))
@sc.param(Int(key="seed"))
@sc.param(Int(key="windowSize", default=5))
@sc.param(Int(key="maxSentenceLength", default=1000))
def SPWord2Vec(context):
    args = context.args

    transorfomer = Word2Vec(
        inputCol=args.inputCol, outputCol=args.outputCol, **args.params
    )
    output = transorfomer.fit(args.inputData).transform(args.inputData)
    return output


if __name__ == "__main__":
    SPWord2Vec()  # pylint: disable=no-value-for-parameter
