# coding=utf-8
from __future__ import print_function

from pyspark.ml.feature import RFormula

from suanpan.arguments import Bool, ListOfString, String
from suanpan.spark import SparkComponent as sc
from suanpan.spark.arguments import HiveTable


@sc.input(HiveTable(key="inputData", table="inputTable", partition="inputPartition"))
@sc.output(
    HiveTable(key="outputData", table="outputTable", partition="outputPartition")
)
@sc.column(String(key="labelColumn", default="label"))
@sc.column(ListOfString(key="selectColumns", required=True))
@sc.column(String(key="featuresCol", default="features"))
@sc.column(String(key="labelCol", default="label"))
@sc.param(Bool(key="forceIndexLabel", default=False))
@sc.param(String(key="stringIndexerOrderType", default="frequencyDesc"))
@sc.param(String(key="handleInvalid", default="error"))
def SPRFormula(context):
    args = context.args

    transorfomer = RFormula(
        formula="{0} ~ {1}".format(args.labelColumn, "+".join(args.selectColumns)),
        featuresCol=args.featuresCol,
        labelCol=args.labelCol,
        **args.params
    )
    output = transorfomer.fit(args.inputData).transform(args.inputData)
    return output


if __name__ == "__main__":
    SPRFormula()  # pylint: disable=no-value-for-parameter
