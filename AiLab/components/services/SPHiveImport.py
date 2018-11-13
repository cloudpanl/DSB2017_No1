# coding=utf-8
from __future__ import print_function

import json

from pyhive import hive
from suanpan.arguments import Int, String
from suanpan.services import Service


class SPHiveImportService(Service):

    arguments = [
        String("hive-host", default="localhost"),
        Int("hive-port"),
        String("hive-database", default="default"),
        String("hive-username"),
        String("hive-password"),
        String("hive-auth"),
    ]

    def call(self, request, context):
        inputData = json.loads(request.in1)
        database, table, columns, data, mode = [
            inputData[i] for i in ("database", "table", "columns", "data", "mode")
        ]
        overwrite = mode == "overwrite"
        with self.connectHive().cursor() as cursor:
            self.useDatabase(cursor, database)
            if overwrite:
                self.dropTable(cursor, table)
            self.createTable(cursor, table, columns)
            self.insertData(cursor, table, data)
            return dict(out1="accepted!")

    def connectHive(self):
        return hive.connect(
            host=self.args.hive_host,
            port=self.args.hive_port,
            database=self.args.hive_database,
            username=self.args.hive_username,
            password=self.args.hive_password,
            auth=self.args.hive_auth,
        )

    def createTable(self, cursor, table, columns):
        cursor.execute(
            "create table if not exists {table} ({columns})".format(
                table=table,
                columns=", ".join(
                    ["{} {}".format(c["name"], c["type"]) for c in columns]
                ),
            )
        )

    def dropTable(self, cursor, table):
        cursor.execute("drop table {}".format(table))

    def insertData(self, cursor, table, data):
        valueGroup = lambda d: "({})".format(", ".join([str(i) for i in d]))
        cursor.execute(
            "insert into {table} values {values}".format(
                table=table, values=", ".join([valueGroup(d) for d in data])
            )
        )

    def useDatabase(self, cursor, database):
        cursor.execute("use {}".format(database))


if __name__ == "__main__":
    SPHiveImportService().start()
