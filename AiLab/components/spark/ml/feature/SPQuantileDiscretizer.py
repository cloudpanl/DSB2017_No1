# coding=utf-8
from __future__ import print_function

from pyspark.ml.feature import QuantileDiscretizer

from suanpan.arguments import Float, Int, String
from suanpan.spark import SparkComponent as sc
from suanpan.spark.arguments import HiveTable


@sc.input(HiveTable(key="inputData", table="inputTable", partition="inputPartition"))
@sc.output(
    HiveTable(key="outputData", table="outputTable", partition="outputPartition")
)
@sc.column(String(key="inputCol", required=True))
@sc.column(String(key="outputCol", required=True))
@sc.param(Int(key="numBuckets", default=2))
@sc.param(Float(key="relativeError", default=0.001))
@sc.param(String(key="handleInvalid", default="error"))
def SPQuantileDiscretizer(context):
    args = context.args

    transorfomer = QuantileDiscretizer(
        inputCol=args.inputCol, outputCol=args.outputCol, **args.params
    )
    output = transorfomer.fit(args.inputData).transform(args.inputData)
    return output


if __name__ == "__main__":
    SPQuantileDiscretizer()  # pylint: disable=no-value-for-parameter
