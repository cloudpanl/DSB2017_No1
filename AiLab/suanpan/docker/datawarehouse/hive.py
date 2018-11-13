# coding=utf-8
from __future__ import print_function

import math
from pyhive import hive

import pandas as pd
from suanpan.log import logger


class HiveDataWarehouse(object):

    DEFAULT_HIVE_DB_NAME = "default"
    DTYPE_SQLTYPE_MAPPING = {
        "int8": "tinyint",
        "int16": "smallint",
        "int32": "int",
        "int64": "bigint",
        "float32": "float",
        "float64": "double",
        "class": "string",
        "datetime": "timestamp",
    }

    def __init__(
        self,
        hiveHost,
        hivePort=10000,
        hiveDatabase="default",
        hiveUsername=None,
        hivePassword=None,
        hiveAuth=None,
        **kwargs
    ):
        self.host = hiveHost
        self.port = hivePort
        self.database = hiveDatabase
        self.username = hiveUsername
        self.password = hivePassword
        self.auth = hiveAuth

        self._connection = None

        self.connect(
            host=self.host,
            port=self.port,
            database=self.database,
            username=self.username,
            password=self.password,
            auth=self.auth,
        )

    @property
    def connection(self):
        if not self._connection:
            raise Exception("Unconnect Error.")
        return self._connection

    def connect(self, *args, **kwargs):
        self.safeClose()
        connection = hive.Connection(*args, **kwargs)
        self.testConnection(connection)
        self._connection = connection
        return self.connection

    def testConnection(self, connection):
        with connection.cursor() as cursor:
            self.selectCurrentTimestamp(cursor)
        logger.info(u"Hive connected.")

    def safeClose(self, *args, **kwargs):
        if self._connection:
            self.close(*args, **kwargs)

    def close(self, *args, **kwargs):
        self.connection.close(*args, **kwargs)
        self._connection = None

    def cursor(self, *args, **kwargs):
        return self.connection.cursor(*args, **kwargs)

    def readTable(
        self, table, partition=None, database=DEFAULT_HIVE_DB_NAME
    ):
        sql = u"select * from {}.{}".format(database, table)

        if partition and partition.strip():
            conditions = u" and ".join(x.strip() for x in partition.split(","))
            sql = u"{} where {}".format(sql, conditions)

        data = pd.read_sql(sql, self.connection)
        data = data.rename(index=str, columns=lambda x: x.split(".")[-1])
        return data

    def writeTable(
        self, table, data, database=DEFAULT_HIVE_DB_NAME, overwrite=True
    ):
        columns = self.getColumns(data)
        with self.cursor() as cursor:
            self.useDatabase(cursor, database)
            if overwrite:
                self.dropTable(cursor, table)
            self.createTable(cursor, table, columns)
            self.insertData(cursor, table, data)

    def execute(self, sql):
        with self.cursor() as cursor:
            return self._execute(cursor, sql)

    def _execute(self, cursor, sql):
        try:
            return cursor.execute(sql)
        except Exception as e:
            logger.error("SQL: {}".format(sql))
            raise e

    def createTable(self, cursor, table, columns):
        self._execute(
            cursor,
            u"create table if not exists {table} ({columns})".format(
                table=table,
                columns=u", ".join(
                    [u"{} {}".format(c["name"], c["type"]) for c in columns]
                ),
            )
        )

    def dropTable(self, cursor, table):
        self._execute(cursor, u"drop table {}".format(table))

    def insertData(self, cursor, table, data, mode="into", per=10000):
        dataLength = len(data)
        logger.info(u"Insert into tabel {}: total {}".format(table, dataLength))
        pieces = dataLength // per + 1
        for i in range(pieces):
            begin = per * i
            end = per * (i + 1)
            values = data[begin:end]
            self.insertPiece(cursor, table, values, mode=mode, begin=begin)

    def insertPiece(self, cursor, table, data, mode="into", begin=0):
        if not data.empty:
            end = begin + len(data)
            logger.info(u"[{} - {}] inserting...".format(begin, end))
            valueGroup = lambda d: u"({})".format(
                u", ".join([u'"{}"'.format(i) for i in d])
            )
            self._execute(
                cursor,
                u"insert {mode} {table} values {values}".format(
                    table=table,
                    values=u", ".join(data.apply(valueGroup, axis=1).values),
                    mode=mode,
                )
            )
            logger.info(u"[{} - {}] Done.".format(begin, end))

    def useDatabase(self, cursor, database):
        self._execute(cursor, u"use {}".format(database))

    def selectCurrentTimestamp(self, cursor):
        self._execute(cursor, u"select current_timestamp()")

    def getColumns(self, data):
        return [
            {"name": name.split(".")[-1], "type": self.toSqlType(dtype)}
            for name, dtype in data.dtypes.items()
        ]

    def toSqlType(self, dtype):
        typeName = self.shortenDatetimeType(dtype.name)
        return self.DTYPE_SQLTYPE_MAPPING.get(typeName, "string")

    def shortenDatetimeType(self, typeName):
        datatimeType = "datetime"
        return datatimeType if typeName.startswith(datatimeType) else typeName
