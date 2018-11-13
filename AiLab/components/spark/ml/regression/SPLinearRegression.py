# coding=utf-8
from __future__ import print_function

from pyspark.ml import Pipeline
from pyspark.ml.feature import RFormula
from pyspark.ml.regression import LinearRegression

from suanpan.arguments import Bool, Float, Int, ListOfString, String
from suanpan.spark import SparkComponent as sc
from suanpan.spark.arguments import HiveTable, SparkModel


@sc.input(HiveTable(key="inputData", table="inputTable", partition="inputPartition"))
@sc.output(SparkModel(key="outputModel"))
@sc.column(String(key="labelColumn", default="label"))
@sc.column(ListOfString(key="selectColumns", required=True))
@sc.param(Int(key="maxIter", default=100))
@sc.param(Float(key="regParam", default=0.0))
@sc.param(Float(key="elasticNetParam", default=0.0))
@sc.param(Float(key="tol", default=1e-06))
@sc.param(Bool(key="fitIntercept", default=True))
@sc.param(Bool(key="standardization", default=True))
@sc.param(String(key="solver", default="auto"))
@sc.param(Int(key="aggregationDepth", default=2))
@sc.param(String(key="loss", default="squaredError"))
@sc.param(Float(key="epsilon", default=1.35))
def SPLinearRegression(context):
    args = context.args

    formula = RFormula(
        formula="{0} ~ {1}".format(args.labelColumn, "+".join(args.selectColumns))
    )
    classifier = LinearRegression(**args.params)
    pipeline = Pipeline(stages=[formula, classifier])
    pipelineModel = pipeline.fit(args.inputData)
    return dict(model=pipelineModel, data=args.inputData, classifier=classifier)


if __name__ == "__main__":
    SPLinearRegression()  # pylint: disable=no-value-for-parameter
