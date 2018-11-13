# coding=utf-8
from __future__ import print_function

from pyspark.ml.feature import MinMaxScaler

from suanpan.arguments import Float, String
from suanpan.spark import SparkComponent as sc
from suanpan.spark.arguments import HiveTable


@sc.input(HiveTable(key="inputData", table="inputTable", partition="inputPartition"))
@sc.output(
    HiveTable(key="outputData", table="outputTable", partition="outputPartition")
)
@sc.column(String(key="inputCol", required=True))
@sc.column(String(key="outputCol", required=True))
@sc.param(Float(key="min", default=0.0))
@sc.param(Float(key="max", default=1.0))
def SPMinMaxScaler(context):
    args = context.args

    transorfomer = MinMaxScaler(
        inputCol=args.inputCol, outputCol=args.outputCol, **args.params
    )
    output = transorfomer.fit(args.inputData).transform(args.inputData)
    return output


if __name__ == "__main__":
    SPMinMaxScaler()  # pylint: disable=no-value-for-parameter
