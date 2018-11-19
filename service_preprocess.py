# coding=utf-8
from __future__ import print_function

import os
import tempfile

from suanpan import asyncio
from suanpan.docker.io import storage

from dsb.preprocessing import full_prep
from service import DSBService


class ServiceProprocess(DSBService):
    def call(self, request, context):
        ossDataFolder = request.in1
        localDataFolder = storage.download(
            ossDataFolder, storage.getPathInTempStore(ossDataFolder)
        )
        ossResultFolder = "majik_test/dsb3/service/preprocess"
        localResultFolder = storage.getPathInTempStore(ossResultFolder)

        full_prep(
            localDataFolder,
            localResultFolder,
            n_worker=asyncio.WORKERS,
            use_existing=False,
        )

        storage.upload(ossResultFolder, localResultFolder)
        return dict(out1=ossResultFolder)


if __name__ == "__main__":
    ServiceProprocess().start()
