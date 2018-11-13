# coding=utf-8
from __future__ import print_function

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import RFormula

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
@sc.param(Int(key="maxIter", default=100))
@sc.param(Float(key="regParam", default=0.0))
@sc.param(Float(key="elasticNetParam", default=0.0))
@sc.param(Float(key="tol", default=1e-06))
@sc.param(Bool(key="fitIntercept", default=True))
@sc.param(Float(key="threshold", default=0.5))
@sc.param(ListOfFloat(key="thresholds"))
@sc.param(Bool(key="standardization", default=True))
@sc.param(String(key="weightCol"))
@sc.param(Int(key="aggregationDepth", default=2))
@sc.param(String(key="family", default="auto"))
# @sc.param(String(key='lowerBoundsOnCoefficients'))
# @sc.param(String(key='upperBoundsOnCoefficients'))
# @sc.param(String(key='lowerBoundsOnIntercepts'))
# @sc.param(String(key='upperBoundsOnIntercepts'))
def SPLogisticRegression(context):
    args = context.args

    formula = RFormula(
        formula="{0} ~ {1}".format(args.labelColumn, "+".join(args.selectColumns))
    )
    classifier = LogisticRegression(**args.params)
    pipeline = Pipeline(stages=[formula, classifier])
    pipelineModel = pipeline.fit(args.inputData)
    return dict(model=pipelineModel, data=args.inputData, classifier=classifier)


if __name__ == "__main__":
    SPLogisticRegression()  # pylint: disable=no-value-for-parameter
