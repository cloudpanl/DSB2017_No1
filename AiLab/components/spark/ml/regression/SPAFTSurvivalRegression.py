# coding=utf-8
from __future__ import print_function

from pyspark.ml import Pipeline
from pyspark.ml.feature import RFormula
from pyspark.ml.regression import AFTSurvivalRegression

from suanpan.arguments import (
    Bool,
    Float,
    Int,
    ListOfFloat,
    ListOfString,
    String,
)
from suanpan.spark import SparkComponent as sc
from suanpan.spark.arguments import HiveTable, SparkModel


@sc.input(HiveTable(key="inputData", table="inputTable", partition="inputPartition"))
@sc.output(SparkModel(key="outputModel"))
@sc.column(String(key="labelColumn", default="label"))
@sc.column(ListOfString(key="selectColumns", required=True))
@sc.param(Bool(key="fitIntercept", default=True))
@sc.param(Int(key="maxIter", default=100))
@sc.param(Float(key="tol", default=1e-06))
@sc.param(String(key="censorCol", default="censor"))
@sc.param(
    ListOfFloat(
        key="quantileProbabilities",
        default=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99],
    )
)
@sc.param(Int(key="aggregationDepth", default=2))
def SPAFTSurvivalRegression(context):
    args = context.args

    formula = RFormula(
        formula="{0} ~ {1}".format(args.labelColumn, "+".join(args.selectColumns))
    )
    classifier = AFTSurvivalRegression(**args.params)
    pipeline = Pipeline(stages=[formula, classifier])
    pipelineModel = pipeline.fit(args.inputData)
    return dict(model=pipelineModel, data=args.inputData, classifier=classifier)


if __name__ == "__main__":
    SPAFTSurvivalRegression()  # pylint: disable=no-value-for-parameter
