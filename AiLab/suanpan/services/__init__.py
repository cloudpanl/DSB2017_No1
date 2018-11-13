# coding=utf-8
from __future__ import print_function

import argparse
import itertools
import time
import traceback
import uuid
from concurrent import futures

import grpc
from suanpan.arguments import Int
from suanpan.log import logger
from suanpan.services import common_pb2, common_pb2_grpc


class Service(common_pb2_grpc.CommonServicer):

    defaultServiceCall = "call"
    defaultArguments = [Int("port", default=8980), Int("workers", default=1)]
    arguments = []

    def __init__(self):
        logger.setLogger(self.name)
        self.args = self.parseArgs()

    @property
    def name(self):
        return self.__class__.__name__

    def generateRequestId(self):
        return uuid.uuid4().hex

    def formatMessage(self, request, msg):
        return "{} - {} - {}".format(
            request.id, request.type or self.defaultServiceCall, msg
        )

    def predict(self, request, context):
        try:
            logger.info(self.formatMessage(request, msg="Start"))
            callFunction = self.getCallFunction(request, context)
            outputs = callFunction(request, context) or {}
            result = dict(success=True, **outputs)
            logger.info(self.formatMessage(request, msg="Done"))
        except:
            tracebackInfo = traceback.format_exc()
            result = dict(success=False, msg=tracebackInfo)
            logger.error(self.formatMessage(request, msg=tracebackInfo))
        finally:
            return common_pb2.Response(request_id=request.id, **result)

    def getCallFunction(self, request, context):
        serviceCall = request.type or self.defaultServiceCall
        callFunction = getattr(self, serviceCall, None)
        if not callFunction:
            raise Exception(
                "Unknown service call: {}.{}".format(self.name, serviceCall)
            )
        return callFunction

    def call(self, request, context, *args):
        pass
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def parseArgs(self):
        parser = argparse.ArgumentParser(description="")
        for arg in itertools.chain(self.defaultArguments, self.arguments):
            arg.addParserArguments(parser)
        return parser.parse_args()

    def start(self):
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=self.args.workers))
        common_pb2_grpc.add_CommonServicer_to_server(self, server)
        server.add_insecure_port("[::]:{}".format(self.args.port))
        server.start()
        logger.info("{} started!".format(self.name))
        try:
            _ONE_DAY_IN_SECONDS = 60 * 60 * 24
            while True:
                time.sleep(_ONE_DAY_IN_SECONDS)
        except KeyboardInterrupt:
            server.stop(0)
