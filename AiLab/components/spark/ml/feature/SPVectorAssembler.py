# coding=utf-8
from __future__ import print_function

from pyspark.ml import Pipeline
from pyspark.ml.feature import RFormula, VectorAssembler

from suanpan.arguments import ListOfString, String
from suanpan.spark import SparkComponent as sc
from suanpan.spark.arguments import HiveTable


@sc.input(HiveTable(key="inputData", table="inputTable", partition="inputPartition"))
@sc.output(
    HiveTable(key="outputData", table="outputTable", partition="outputPartition")
)
@sc.column(ListOfString(key="inputCols", required=True))
@sc.column(String(key="outputCol", required=True))
def SPVectorAssembler(context):
    args = context.args

    transorfomer = VectorAssembler(**args.columns)
    output = transorfomer.transform(args.inputData)
    return output


if __name__ == "__main__":
    SPVectorAssembler()  # pylint: disable=no-value-for-parameter
