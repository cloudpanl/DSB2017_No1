# coding=utf-8
from __future__ import print_function

from pyspark.ml.evaluation import ClusteringEvaluator

from suanpan.arguments import String
from suanpan.spark import SparkComponent as sc
from suanpan.spark.arguments import HiveTable


@sc.input(HiveTable(key="inputData", table="inputTable", partition="inputPartition"))
@sc.output(
    HiveTable(key="evaluateData", table="outputTable", partition="outputPartition")
)
@sc.param(String(key="predictionCol", default="prediction"))
@sc.param(String(key="featuresCol", default="features"))
def SPClusteringEvaluator(context):
    spark = context.spark
    args = context.args

    evaluator = ClusteringEvaluator(**args.params)
    metricNames = ["silhouette"]
    evaluateValues = [
        evaluator.evaluate(args.inputData, {evaluator.metricName: name})
        for name in metricNames
    ]
    evaluateData = spark.createDataFrame([evaluateValues], metricNames)
    return evaluateData


if __name__ == "__main__":
    SPClusteringEvaluator()  # pylint: disable=no-value-for-parameter
