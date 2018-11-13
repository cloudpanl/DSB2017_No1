# coding=utf-8
from __future__ import print_function

import tempfile

from suanpan.docker import DockerComponent
from suanpan.arguments import String, Int


class ToolComponent(DockerComponent):
    dataWarehouseArguments = [
        String("dw-type", default="hive"),
        String("dw-hive-host", default="47.94.82.175"),
        Int("dw-hive-port", default=10000),
        String("dw-hive-database", default="default"),
        String("dw-hive-username", default="spark"),
        String("dw-hive-password"),
        String("dw-hive-auth"),
    ]
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
    defaultArguments = dataWarehouseArguments + storageArguments
