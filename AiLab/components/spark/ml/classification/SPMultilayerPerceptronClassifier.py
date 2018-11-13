# coding=utf-8
from __future__ import print_function

from pyspark.ml import Pipeline
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.feature import RFormula

from suanpan.arguments import Float, Int, ListOfInt, ListOfString, String
from suanpan.spark import SparkComponent as sc
from suanpan.spark.arguments import HiveTable, SparkModel


@sc.input(HiveTable(key="inputData", table="inputTable", partition="inputPartition"))
@sc.output(SparkModel(key="outputModel"))
@sc.column(String(key="labelColumn", default="label"))
@sc.column(ListOfString(key="selectColumns", required=True))
@sc.param(Int(key="maxIter", default=100))
@sc.param(Float(key="tol", default=1e-06))
@sc.param(Int(key="seed"))
@sc.param(ListOfInt(key="layers", required=True))
@sc.param(Int(key="blockSize", default=128))
@sc.param(Float(key="stepSize", default=0.03))
@sc.param(String(key="solver", default="l-bfgs"))
@sc.param(ListOfInt(key="initialWeights"))
def SPMultilayerPerceptronClassifier(context):
    args = context.args

    formula = RFormula(
        formula="{0} ~ {1}".format(args.labelColumn, "+".join(args.selectColumns))
    )
    classifier = MultilayerPerceptronClassifier(**args.params)
    pipeline = Pipeline(stages=[formula, classifier])
    pipelineModel = pipeline.fit(args.inputData)
    return dict(model=pipelineModel, data=args.inputData, classifier=classifier)


if __name__ == "__main__":
    SPMultilayerPerceptronClassifier()  # pylint: disable=no-value-for-parameter
