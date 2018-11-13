# coding=utf-8
from __future__ import print_function

from pyspark.ml.feature import ChiSqSelector

from suanpan.arguments import Float, Int, ListOfString, String
from suanpan.spark import SparkComponent as sc
from suanpan.spark.arguments import HiveTable


@sc.input(HiveTable(key="inputData", table="inputTable", partition="inputPartition"))
@sc.output(
    HiveTable(key="outputData", table="outputTable", partition="outputPartition")
)
@sc.column(String(key="featuresCol", default="features"))
@sc.column(String(key="outputCol", required=True))
@sc.column(String(key="labelCol", default="label"))
@sc.param(Int(key="numTopFeatures", default=50))
@sc.param(String(key="selectorType", default="numTopFeatures"))
@sc.param(Float(key="percentile", default=0.1))
@sc.param(Float(key="fpr", default=0.05))
@sc.param(Float(key="fdr", default=0.05))
@sc.param(Float(key="fwe", default=0.05))
def SPChiSqSelector(context):
    args = context.args

    transorfomer = ChiSqSelector(
        featuresCol=args.featuresCol,
        outputCol=args.outputCol,
        labelCol=args.labelCol,
        **args.params
    )
    output = transorfomer.fit(args.inputData).transform(args.inputData)
    return output


if __name__ == "__main__":
    SPChiSqSelector()  # pylint: disable=no-value-for-parameter
