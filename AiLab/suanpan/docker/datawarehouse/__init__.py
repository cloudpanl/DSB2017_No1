# coding=utf-8
from __future__ import print_function

from suanpan.docker.datawarehouse.hive import HiveDataWarehouse
from suanpan.proxy import Proxy


class DataWarehouseProxy(Proxy):
    MAPPING = {"hive": HiveDataWarehouse}


dw = DataWarehouseProxy()
