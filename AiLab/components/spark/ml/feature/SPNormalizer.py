# coding=utf-8
from __future__ import print_function

from pyspark.ml.feature import Normalizer

from suanpan.arguments import Float, String
from suanpan.spark import SparkComponent as sc
from suanpan.spark.arguments import HiveTable


@sc.input(HiveTable(key="inputData", table="inputTable", partition="inputPartition"))
@sc.output(
    HiveTable(key="outputData", table="outputTable", partition="outputPartition")
)
@sc.column(String(key="inputCol", required=True))
@sc.column(String(key="outputCol", required=True))
@sc.param(Float(key="p", default=2.0))
def SPNormalizer(context):
    args = context.args

    transorfomer = Normalizer(
        inputCol=args.inputCol, outputCol=args.outputCol, **args.params
    )
    output = transorfomer.transform(args.inputData)
    return output


if __name__ == "__main__":
    SPNormalizer()  # pylint: disable=no-value-for-parameter
