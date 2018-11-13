# coding=utf-8
from __future__ import print_function

from suanpan.arguments import String
from suanpan.components import Component as c
from suanpan.docker.io import storage


@c.param(String(key="local", required=True))
@c.param(String(key="bucket", default="suanpan"))
@c.param(String(key="remote", required=True))
def upload(context):
    args = context.args

    storage.setBackend(
        type="oss",
        ossAccessId="LTAIgV6cMz4TgHZB",
        ossAccessKey="M6jP8a1KN2kfZR51M08UiEufnzQuiY",
        ossBucketName=args.bucket,
        ossEndpoint="http://oss-cn-beijing.aliyuncs.com",
        ossTempStore="tmp",
    )
    storage.upload(args.remote, args.local)


if __name__ == "__main__":
    upload()  # pylint: disable=no-value-for-parameter
