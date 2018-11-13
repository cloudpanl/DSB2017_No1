# coding=utf-8
from __future__ import print_function

from pyspark.ml.feature import SQLTransformer

from suanpan.arguments import String
from suanpan.spark import SparkComponent as sc
from suanpan.spark.arguments import HiveTable


@sc.input(HiveTable(key='inputData', table='inputTable', partition='inputPartition'))
@sc.output(HiveTable(key='outputData', table='outputTable', partition='outputPartition'))
@sc.param(String(key='statement', required=True))
def SPSQLTransformer(context):
    args = context.args

    transorfomer = SQLTransformer(**args.params)
    output = transorfomer.transform(args.inputData)
    return output


if __name__ == '__main__':
    SPSQLTransformer()  # pylint: disable=no-value-for-parameter
