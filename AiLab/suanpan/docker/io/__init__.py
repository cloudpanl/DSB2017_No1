# coding=utf-8
from __future__ import print_function

from suanpan.docker.io.oss import OssStorage
from suanpan.proxy import Proxy


class StorageProxy(Proxy):
    MAPPING = {"oss": OssStorage}


storage = StorageProxy()
