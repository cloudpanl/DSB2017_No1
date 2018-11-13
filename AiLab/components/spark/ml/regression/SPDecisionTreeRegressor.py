# coding=utf-8
from __future__ import print_function

from pyspark.ml import Pipeline
from pyspark.ml.feature import RFormula
from pyspark.ml.regression import DecisionTreeRegressor

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
@sc.param(String(key="impurity", default="variance"))
@sc.param(Int(key="seed"))
@sc.param(String(key="varianceCol"))
def SPDecisionTreeRegressor(context):
    args = context.args

    formula = RFormula(
        formula="{0} ~ {1}".format(args.labelColumn, "+".join(args.selectColumns))
    )
    classifier = DecisionTreeRegressor(**args.params)
    pipeline = Pipeline(stages=[formula, classifier])
    pipelineModel = pipeline.fit(args.inputData)
    return dict(model=pipelineModel, data=args.inputData, classifier=classifier)


if __name__ == "__main__":
    SPDecisionTreeRegressor()  # pylint: disable=no-value-for-parameter
