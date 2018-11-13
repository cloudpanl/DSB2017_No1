# coding=utf-8
from __future__ import print_function

import argparse
import json

from suanpan.log import logger


class Arg(object):
    def __init__(self, key, **kwargs):
        if "default" in kwargs:
            kwargs.update(required=False)

        self.key = key
        self.value = None
        self.type = kwargs.pop("type", str)
        self.kwargs = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def isSet(self):
        return self.kwargs.get("required") or self.kwargs.get("default") != self.value

    def addParserArguments(self, parser):
        parser.add_argument(
            "--{}".format(self.key), type=self.typeDecorator(self.type), **self.kwargs
        )

    def typeDecorator(self, typeFunc):
        def _decorator(value):
            return getattr(self, "default", None) if value == "" else typeFunc(value)

        return _decorator

    def load(self, args):
        self.value = getattr(args, self.key)
        logger.info(
            "({type}) {key} loaded: {value}".format(
                type=self.__class__.__name__, key=self.key, value=self.value
            )
        )
        return self.value

    def format(self, context):
        return self.value

    def save(self, context, result):
        logger.warning(
            "({type}) {key} not support save. Do nothing!".format(
                type=self.__class__.__name__, key=self.key
            )
        )


class String(Arg):
    def __init__(self, key, **kwargs):
        super(String, self).__init__(key, type=str, **kwargs)


class Int(Arg):
    def __init__(self, key, **kwargs):
        super(Int, self).__init__(key, type=int, **kwargs)


class Float(Arg):
    def __init__(self, key, **kwargs):
        super(Float, self).__init__(key, type=float, **kwargs)


class Bool(Arg):
    def __init__(self, key, **kwargs):
        kwargs.update(default=False)
        super(Bool, self).__init__(key, type=type(self).str2bool, **kwargs)

    @classmethod
    def str2bool(cls, string):
        if string.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif string.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")


class List(Arg):
    def __init__(self, key, **kwargs):
        super(List, self).__init__(key, type=type(self).str2list, **kwargs)

    @classmethod
    def str2list(cls, string):
        try:
            return [cls.transform(i.strip()) for i in string.split(",") if i.strip()]
        except:
            raise argparse.ArgumentTypeError("{} value expected.".format(cls.__name__))

    @classmethod
    def transform(cls, item):
        return item


class ListOfString(List):
    pass


class ListOfInt(List):
    @classmethod
    def transform(cls, item):
        return int(item)


class ListOfFloat(List):
    @classmethod
    def transform(cls, item):
        return float(item)


class ListOfBool(List):
    @classmethod
    def transform(cls, item):
        return Bool.str2bool(item)
