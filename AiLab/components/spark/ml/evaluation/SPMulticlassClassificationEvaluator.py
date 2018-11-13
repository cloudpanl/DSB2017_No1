# coding=utf-8
from __future__ import print_function

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from suanpan.arguments import String
from suanpan.spark import SparkComponent as sc
from suanpan.spark.arguments import HiveTable


@sc.input(HiveTable(key="inputData", table="inputTable", partition="inputPartition"))
@sc.output(
    HiveTable(key="evaluateData", table="outputTable", partition="outputPartition")
)
@sc.param(String(key="predictionCol", default="prediction"))
@sc.param(String(key="labelCol", default="label"))
def SPMulticlassClassificationEvaluator(context):
    spark = context.spark
    args = context.args

    evaluator = MulticlassClassificationEvaluator(**args.params)
    metricNames = ["f1", "weightedPrecision", "weightedRecall", "accuracy"]
    evaluateValues = [
        evaluator.evaluate(args.inputData, {evaluator.metricName: name})
        for name in metricNames
    ]
    evaluateData = spark.createDataFrame([evaluateValues], metricNames)
    return evaluateData


if __name__ == "__main__":
    SPMulticlassClassificationEvaluator()  # pylint: disable=no-value-for-parameter
