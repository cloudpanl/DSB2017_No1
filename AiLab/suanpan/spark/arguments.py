# coding=utf-8
from __future__ import print_function

import json

from pyspark2pmml import PMMLBuilder
from pyspark.ml import Pipeline, PipelineModel
from suanpan.arguments import Arg, Bool, Int, ListOfString, String
from suanpan.spark import db, io


class HiveTable(Arg):
    def __init__(self, key, table, partition, sortColumns=None):
        super(HiveTable, self).__init__(key)
        sortColumns = sortColumns or "{}SortColumns".format(table)
        self.table = String(key=table, required=True)
        self.partition = String(key=partition)
        self.sortColumns = ListOfString(key=sortColumns, default=[])

    def addParserArguments(self, parser):
        self.table.addParserArguments(parser)
        self.partition.addParserArguments(parser)
        self.sortColumns.addParserArguments(parser)

    def load(self, args):
        self.table.load(args)
        self.partition.load(args)
        self.sortColumns.load(args)
        self.value = dict(
            table=self.table.value,
            partition=self.partition.value,
            sortColumns=self.sortColumns.value,
        )

    def format(self, context):
        self.value = db.readTable(context.spark, self.table.value, self.partition.value)
        self.value = self.value.sort(self.sortColumns.value)
        return self.value

    def save(self, context, result):
        data = result.value
        db.writeTable(context.spark, self.table.value, data)


class SparkModel(Arg):
    def __init__(self, key, **kwargs):
        kwargs.update(required=True)
        super(SparkModel, self).__init__(key, **kwargs)

    def format(self, context):
        self.value = PipelineModel.load(io.getStoragePath(context.spark, self.value))
        return self.value

    def save(self, context, result):
        spark = context.spark

        modelPath = io.getStoragePath(spark, self.value)
        pmmlPath = modelPath + "/pmml"

        result.model.write().overwrite().save(modelPath)
        with io.open(spark, pmmlPath, mode="w") as file:
            pmmlBuilder = PMMLBuilder(
                spark.sparkContext, result.data, result.model
            ).putOption(result.classifier, "compact", True)
            pmml = pmmlBuilder.buildByteArray()
            file.write(pmml)


class PmmlModel(Arg):
    def __init__(self, key, **kwargs):
        kwargs.update(required=True)
        super(PmmlModel, self).__init__(key, **kwargs)

    def format(self, context):
        spark = context.spark

        pmmlPath = self.pmml_path(spark)
        with io.open(spark, pmmlPath, mode="r") as file:
            self.value = file.read()

        return self.value

    def pmml_path(self, spark):
        modelPath = io.getStoragePath(spark, self.value)
        pmmlPath = modelPath + "/pmml"
        return pmmlPath


class OdpsTable(Arg):
    def __init__(
        self,
        key,
        accessId,
        accessKey,
        odpsUrl,
        tunnelUrl,
        project,
        table,
        partition,
        overwrite,
        numPartitions,
    ):
        super(OdpsTable, self).__init__(key)
        self.accessId = String(key=accessId, required=True)
        self.accessKey = String(key=accessKey, required=True)
        self.odpsUrl = String(key=odpsUrl, required=True)
        self.tunnelUrl = String(key=tunnelUrl, required=True)
        self.project = String(key=project, required=True)
        self.table = String(key=table, required=True)
        self.partition = String(key=partition)
        self.overwrite = Bool(key=overwrite, default=False)
        self.numPartitions = Int(key=numPartitions, default=2)

    def addParserArguments(self, parser):
        self.accessId.addParserArguments(parser)
        self.accessKey.addParserArguments(parser)
        self.odpsUrl.addParserArguments(parser)
        self.tunnelUrl.addParserArguments(parser)
        self.project.addParserArguments(parser)
        self.table.addParserArguments(parser)
        self.partition.addParserArguments(parser)
        self.overwrite.addParserArguments(parser)
        self.numPartitions.addParserArguments(parser)

    def load(self, args):
        self.accessId.load(args)
        self.accessKey.load(args)
        self.odpsUrl.load(args)
        self.tunnelUrl.load(args)
        self.project.load(args)
        self.table.load(args)
        self.partition.load(args)
        self.overwrite.load(args)
        self.numPartitions.load(args)
        self.value = dict(
            accessId=self.accessId.value,
            accessKey=self.accessKey.value,
            odpsUrl=self.odpsUrl.value,
            tunnelUrl=self.tunnelUrl.value,
            table=self.table.value,
            partition=self.partition.value,
            overwrite=self.overwrite.value,
            numPartitions=self.numPartitions.value,
        )

    def format(self, context):
        self.value = db.readOdpsTable(
            context.spark,
            accessId=self.accessId.value,
            accessKey=self.accessKey.value,
            odpsUrl=self.odpsUrl.value,
            tunnelUrl=self.tunnelUrl.value,
            project=self.project.value,
            table=self.table.value,
            partition=self.partition.value,
            numPartitions=self.numPartitions.value,
        )
        return self.value

    def save(self, context, result):
        db.writeOdpsTable(
            context.spark,
            accessId=self.accessId.value,
            accessKey=self.accessKey.value,
            odpsUrl=self.odpsUrl.value,
            tunnelUrl=self.tunnelUrl.value,
            project=self.project.value,
            table=self.table.value,
            data=result.value,
            partition=self.partition.value,
            overwrite=self.overwrite.value,
        )


class Visual(Arg):
    def __init__(self, key, **kwargs):
        kwargs.update(required=True)
        super(Visual, self).__init__(key, **kwargs)

    def save(self, context, result):
        spark = context.spark

        visualPath = io.getStoragePath(spark, self.value) + "/part-00000"
        with io.open(spark, visualPath, mode="w") as file:
            file.write(result.value)


class Json(Arg):
    def format(self, context):
        spark = context.spark

        jsonPath = self.json_path(spark)
        with io.open(spark, jsonPath, mode="r") as file:
            self.value = json.load(file)
            return self.value

    def save(self, context, result):
        spark = context.spark

        data = (
            result.value.to_dict()
            if self.only_has_value(result) and isinstance(result.value, dict)
            else result.to_dict()
        )
        jsonPath = self.json_path(spark)
        with io.open(spark, jsonPath, mode="w") as file:
            json.dump(data, file)

    def json_path(self, spark):
        dataPath = io.getStoragePath(spark, self.value)
        jsonPath = dataPath + ".json"
        return jsonPath

    def only_has_value(self, result):
        return "value" in result and len(result) == 1
