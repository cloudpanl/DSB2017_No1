# coding=utf-8
from __future__ import print_function

import tempfile
import traceback
from contextlib import contextmanager

from pyhive import hive

from suanpan.arguments import Int, String
from suanpan.components import Component, Context
from suanpan.docker.datawarehouse import dw
from suanpan.docker.io import storage
from suanpan.log import logger


class DockerComponent(Component):

    dataWarehouseArguments = [
        String("dw-type", default="hive"),
        String("dw-hive-host", default="localhost"),
        Int("dw-hive-port"),
        String("dw-hive-database", default="default"),
        String("dw-hive-username"),
        String("dw-hive-password"),
        String("dw-hive-auth"),
    ]
    storageArguments = [
        String("storage-type", default="oss"),
        String("storage-oss-access-id"),
        String("storage-oss-access-key"),
        String("storage-oss-bucket-name", default="suanpan"),
        String("storage-oss-endpoint", default="http://oss-cn-beijing.aliyuncs.com"),
        String("storage-oss-delimiter", default="/"),
        String("storage-oss-temp-store", default=tempfile.gettempdir()),
        Int("storage-oss-download-num-threads", default=1),
        String("storage-oss-download-store-name", default=".py-oss-download"),
        Int("storage-oss-upload-num-threads", default=1),
        String("storage-oss-upload-store-name", default=".py-oss-upload"),
    ]
    defaultArguments = dataWarehouseArguments + storageArguments

    @contextmanager
    def context(self, args):
        self.storage = self.setStorage(args)
        self.dw = self.setDataWarehouse(args)
        yield Context(dw=self.dw, storage=self.storage)
        self.dw.close()

    def setDataWarehouse(self, args):
        return dw.setBackend(
            **self.defaultArgumentsFormat(args, self.dataWarehouseArguments)
        )

    def setStorage(self, args):
        return storage.setBackend(
            **self.defaultArgumentsFormat(args, self.storageArguments)
        )

    def defaultArgumentsFormat(self, args, arguments):
        arguments = (arg.key.replace("-", "_") for arg in arguments)
        return {
            self.defaultArgumentKeyFormat(arg): getattr(args, arg) for arg in arguments
        }

    def defaultArgumentKeyFormat(self, key):
        return self.toCamelCase(self.removePrefix(key))

    def removePrefix(self, string, delimiter="_", num=1):
        pieces = string.split(delimiter)
        pieces = pieces[num:] if len(pieces) > num else pieces
        return delimiter.join(pieces)

    def toCamelCase(self, string, delimiter="_"):
        camelCaseUpper = lambda i, s: s[0].upper() + s[1:] if i and s else s
        return "".join(
            [camelCaseUpper(i, s) for i, s in enumerate(string.split(delimiter))]
        )
