# coding=utf-8
from __future__ import print_function

from pyspark.ml import Pipeline
from pyspark.ml.classification import OneVsRest
from pyspark.ml.feature import RFormula

from suanpan.arguments import Int, ListOfString, String
from suanpan.spark import SparkComponent as sc
from suanpan.spark.arguments import HiveTable, SparkModel


@sc.input(HiveTable(key="inputData", table="inputTable", partition="inputPartition"))
@sc.output(SparkModel(key="outputModel"))
@sc.column(String(key="labelColumn", default="label"))
@sc.column(ListOfString(key="selectColumns", required=True))
# @sc.param(String(key='classifier'))
@sc.param(String(key="weightCol"))
@sc.param(Int(key="parallelism", default=1))
def SPOneVsRest(context):
    args = context.args

    formula = RFormula(
        formula="{0} ~ {1}".format(args.labelColumn, "+".join(args.selectColumns))
    )
    classifier = OneVsRest(**args.params)
    pipeline = Pipeline(stages=[formula, classifier])
    pipelineModel = pipeline.fit(args.inputData)
    return dict(model=pipelineModel, data=args.inputData, classifier=classifier)


if __name__ == "__main__":
    SPOneVsRest()  # pylint: disable=no-value-for-parameter
