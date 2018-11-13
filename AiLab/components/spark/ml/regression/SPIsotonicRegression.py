# coding=utf-8
from __future__ import print_function

from pyspark.ml import Pipeline
from pyspark.ml.feature import RFormula
from pyspark.ml.regression import IsotonicRegression

from suanpan.arguments import Bool, Int, ListOfString, String
from suanpan.spark import SparkComponent as sc
from suanpan.spark.arguments import HiveTable, SparkModel


@sc.input(HiveTable(key="inputData", table="inputTable", partition="inputPartition"))
@sc.output(SparkModel(key="outputModel"))
@sc.column(String(key="labelColumn", default="label"))
@sc.column(ListOfString(key="selectColumns", required=True))
@sc.param(Bool(key="isotonic", default=True))
@sc.param(Int(key="featureIndex", default=0))
def SPIsotonicRegression(context):
    args = context.args

    formula = RFormula(
        formula="{0} ~ {1}".format(args.labelColumn, "+".join(args.selectColumns))
    )
    classifier = IsotonicRegression(**args.params)
    pipeline = Pipeline(stages=[formula, classifier])
    pipelineModel = pipeline.fit(args.inputData)
    return dict(model=pipelineModel, data=args.inputData, classifier=classifier)


if __name__ == "__main__":
    SPIsotonicRegression()  # pylint: disable=no-value-for-parameter
