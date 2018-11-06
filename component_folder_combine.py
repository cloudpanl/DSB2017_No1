# coding: utf-8
from __future__ import print_function

# from suanpan.docker.arguments import File, Folder, HiveTable
from suanpan import asyncio, convert, path, utils
from suanpan.arguments import Bool, Int, String
from suanpan.components import Component as dc


@dc.input(String(key="inputFolder1", required=True))
@dc.input(String(key="inputFolder2", required=True))
@dc.output(String(key="outputFolder", required=True))
@dc.param(String(key="mode", default="replace"))
def SPFolderCombine(context):
    args = context.args

    return path.merge([args.inputFolder1, args.inputFolder2], dist=args.outputFolder)


if __name__ == "__main__":
    SPFolderCombine()  # pylint: disable=no-value-for-parameter
