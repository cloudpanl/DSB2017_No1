# coding=utf-8
from __future__ import print_function

import copy
import json

import grpc

from suanpan.services import common_pb2, common_pb2_grpc

import_data = {
    "database": "default",
    "table": "majik_test_temp_table",
    "columns": [
        {"name": "fea_0", "type": "double"},
        {"name": "fea_1", "type": "double"},
        {"name": "fea_2", "type": "double"},
        {"name": "fea_3", "type": "double"},
        {"name": "fea_4", "type": "double"},
    ],
    "data": [
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
    ],
    "mode": "append",
}

import_data_overwrite = copy.copy(import_data)
import_data_overwrite.update(
    mode="overwrite",
    data=[
        [10, 2, 3, 4, 5],
        [11, 2, 3, 4, 5],
        [12, 2, 3, 4, 5],
        [13, 2, 3, 4, 5],
        [14, 2, 3, 4, 5],
    ],
)


def data_import(data):
    return rpc_call("localhost:8980", json.dumps(dict(in1=data)))


def rpc_call(target, data):
    with grpc.insecure_channel(target) as channel:
        stub = common_pb2_grpc.CommonStub(channel)
        request = common_pb2.Request(id="id", type="test", data=data)
        response = stub.predict(request)
        print(response)
        if not response.success:
            raise Exception(response.msg)
        return response.data


def main():
    data_import(import_data)
    data_import(import_data_overwrite)
    data_import(import_data)


if __name__ == "__main__":
    main()
