# coding=utf-8
from __future__ import print_function

from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import RFormula

from suanpan.arguments import Bool, Float, Int, ListOfString, String
from suanpan.spark import SparkComponent as sc
from suanpan.spark.arguments import HiveTable, SparkModel


@sc.input(HiveTable(key="inputData", table="inputTable", partition="inputPartition"))
@sc.output(SparkModel(key="outputModel"))
@sc.column(String(key="labelColumn", default="label"))
@sc.column(ListOfString(key="selectColumns", required=True))
@sc.param(Int(key="maxDepth", default=5))
@sc.param(Int(key="maxBins", default=32))
@sc.param(Int(key="minInstancesPerNode", default=1))
@sc.param(Float(key="minInfoGain", default=0.0))
@sc.param(Int(key="maxMemoryInMB", default=256))
@sc.param(Bool(key="cacheNodeIds", default=False))
@sc.param(Int(key="checkpointInterval", default=10))
@sc.param(String(key="impurity", default="gini"))
@sc.param(Int(key="seed"))
def SPDecisionTreeClassifier(context):
    args = context.args

    formula = RFormula(
        formula="{0} ~ {1}".format(args.labelColumn, "+".join(args.selectColumns))
    )
    classifier = DecisionTreeClassifier(**args.params)
    pipeline = Pipeline(stages=[formula, classifier])
    pipelineModel = pipeline.fit(args.inputData)
    return dict(model=pipelineModel, data=args.inputData, classifier=classifier)


if __name__ == "__main__":
    SPDecisionTreeClassifier()  # pylint: disable=no-value-for-parameter
