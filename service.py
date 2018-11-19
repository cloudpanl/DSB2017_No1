# coding=utf-8
from __future__ import print_function

import argparse
import os
import tempfile

from suanpan.arguments import Int, String
from suanpan.docker.io import storage
from suanpan.services import Service


class DSBService(Service):

    storageArguments = [
        String("storage-type", default="oss"),
        String("storage-oss-access-id", default="LTAIgV6cMz4TgHZB"),
        String("storage-oss-access-key", default="M6jP8a1KN2kfZR51M08UiEufnzQuiY"),
        String("storage-oss-bucket-name", default="suanpan"),
        String("storage-oss-endpoint", default="http://oss-cn-beijing.aliyuncs.com"),
        String("storage-oss-delimiter", default="/"),
        String("storage-oss-temp-store", default=tempfile.gettempdir()),
        Int("storage-oss-download-num-threads", default=1),
        String("storage-oss-download-store-name", default=".py-oss-download"),
        Int("storage-oss-upload-num-threads", default=1),
        String("storage-oss-upload-store-name", default=".py-oss-upload"),
    ]

    def parseArguments(self, arguments, description=""):
        parser = argparse.ArgumentParser(description=description)
        for arg in arguments:
            arg.addParserArguments(parser)
        return parser.parse_known_args()[0]

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

    def start(self):
        args = self.parseArguments(self.storageArguments)
        self.setStorage(args)
        super(DSBService, self).start()
