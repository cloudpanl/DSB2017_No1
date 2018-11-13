# coding=utf-8
from __future__ import print_function

from suanpan.arguments import String
from suanpan.components import Component as c
from suanpan.docker.io import storage


@c.param(String(key="path", required=True))
def remove(context):
    args = context.args

    storage.setBackend(
        type="oss",
        ossAccessId="LTAIgV6cMz4TgHZB",
        ossAccessKey="M6jP8a1KN2kfZR51M08UiEufnzQuiY",
        ossBucketName="suanpan",
        ossEndpoint="http://oss-cn-beijing.aliyuncs.com",
        ossTempStore="tmp",
    )
    storage.remove(args.path)


if __name__ == "__main__":
    remove()  # pylint: disable=no-value-for-parameter
