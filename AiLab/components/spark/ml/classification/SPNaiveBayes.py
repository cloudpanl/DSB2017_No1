# coding=utf-8
from __future__ import print_function

from pyspark.ml import Pipeline
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.feature import RFormula

from suanpan.arguments import Float, ListOfFloat, ListOfString, String
from suanpan.spark import SparkComponent as sc
from suanpan.spark.arguments import HiveTable, SparkModel


@sc.input(HiveTable(key="inputData", table="inputTable", partition="inputPartition"))
@sc.output(SparkModel(key="outputModel"))
@sc.column(String(key="labelColumn", default="label"))
@sc.column(ListOfString(key="selectColumns", required=True))
@sc.param(Float(key="smoothing", default=1.0))
@sc.param(String(key="modelType", default="multinomial"))
@sc.param(ListOfFloat(key="thresholds"))
@sc.param(String(key="weightCol"))
def SPNaiveBayes(context):
    args = context.args

    formula = RFormula(
        formula="{0} ~ {1}".format(args.labelColumn, "+".join(args.selectColumns))
    )
    classifier = NaiveBayes(**args.params)
    pipeline = Pipeline(stages=[formula, classifier])
    pipelineModel = pipeline.fit(args.inputData)
    return dict(model=pipelineModel, data=args.inputData, classifier=classifier)


if __name__ == "__main__":
    SPNaiveBayes()  # pylint: disable=no-value-for-parameter
