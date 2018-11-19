# coding=utf-8
from __future__ import print_function

import copy
import json

import grpc

from suanpan.services import common_pb2, common_pb2_grpc


DATA = "majik_test/DSB2017_Data/DSB3/stage1_samples"
CKPT = "majik_test/component_n_net_predict_input_checkpoint/detector/model.ckpt"


def preprocess(data):
    return rpc_call("localhost:8981", in1=data)


def predict(ckpt, data):
    return rpc_call("localhost:8982", in1=ckpt, in2=data)


def dector(data):
    return rpc_call("localhost:8983", in1=data)


def rpc_call(target, **kwargs):
    with grpc.insecure_channel(target) as channel:
        stub = common_pb2_grpc.CommonStub(channel)
        request = common_pb2.Request(id="test_id", **kwargs)
        response = stub.predict(request)
        print(response)
        if not response.success:
            raise Exception(response.msg)
        return response.out1


def main():
    # import pdb; pdb.set_trace()
    preprocessed_data = preprocess(DATA)
    result_data = predict(CKPT, preprocessed_data)
    dector_images = dector(result_data)


if __name__ == "__main__":
    main()
