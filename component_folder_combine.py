# coding: utf-8
from __future__ import print_function

from suanpan import path
from suanpan.arguments import String
from suanpan.docker import DockerComponent as dc
from suanpan.docker.arguments import Folder


@dc.input(Folder(key="inputFolder1", required=True))
@dc.input(Folder(key="inputFolder2", required=True))
@dc.output(Folder(key="outputFolder", required=True))
@dc.param(String(key="mode", default="replace"))
def SPFolderCombine(context):
    args = context.args

    return path.merge([args.inputFolder1, args.inputFolder2], dist=args.outputFolder)


if __name__ == "__main__":
    SPFolderCombine()  # pylint: disable=no-value-for-parameter
