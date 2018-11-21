# coding=utf-8
from __future__ import print_function

import os

from dsb.preprocessing import full_prep
from suanpan import asyncio
from suanpan.services import Handler as h
from suanpan.services import Service
from suanpan.services.arguments import Folder


class ServiceProprocess(Service):

    @h.input(Folder(key="inputDataFolder"))
    @h.output(Folder(key="outputDataFolder"))
    def call(self, context):
        args = context.args

        inputDataFolder = args.inputDataFolder
        outputDataFolder = args.outputDataFolder

        full_prep(
            inputDataFolder,
            outputDataFolder,
            n_worker=asyncio.WORKERS,
            use_existing=False,
        )

        return outputDataFolder


if __name__ == "__main__":
    ServiceProprocess().start()
