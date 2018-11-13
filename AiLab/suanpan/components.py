# coding=utf-8
from __future__ import print_function

import argparse
import contextlib
import copy
import itertools
import traceback
from collections import defaultdict

from suanpan.log import logger


class Context(object):
    def __init__(self, **kwargs):
        self._dict = kwargs
        self.__dict__.update(
            {
                k: self.__class__(**v) if isinstance(v, dict) else v
                for k, v in kwargs.items()
            }
        )

    def __getitem__(self, key):
        return self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def __len__(self):
        return len(self._dict)

    def __str__(self):
        return "<{}: {}>".format(self.__class__.__name__, self._dict)

    def __repr__(self):
        return str(self)

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    def items(self):
        return self._dict.items()

    def to_dict(self):
        return copy.deepcopy(self._dict)


class Arguments(Context):
    pass


class Result(Context):
    def __init__(self, value):
        kwargs = value if isinstance(value, dict) else dict(value=value)
        super(Result, self).__init__(**kwargs)


class Component(object):

    defaultArguments = []

    def __init__(self, funcOrComponent):
        if isinstance(funcOrComponent, Component):
            self.runFunc = funcOrComponent.runFunc
            self.arguments = funcOrComponent.arguments
        else:
            self.runFunc = funcOrComponent
            self.arguments = defaultdict(list)

    def __call__(self, *arg, **kwargs):
        try:
            logger.setLogger(self.name)
            self.run(*arg, **kwargs)
        except Exception as e:
            logger.error(traceback.format_exc())
            raise e

    @property
    def name(self):
        return self.runFunc.__name__

    def run(self, *arg, **kwargs):
        logger.info("Starting...")
        defaultArgs = self.parseDefaultArguments()
        with self.context(defaultArgs) as context:
            args = self.parseArguments(self.getArguments())
            args = self.transformArguments(context, args)
            setattr(context, "args", args)
            results = self.runFunc(context)
            self.saveOutputs(context, results)
        logger.info("Done.")

    @contextlib.contextmanager
    def context(self, args):
        yield Context()

    def parseDefaultArguments(self):
        return self.parseArguments(self.defaultArguments, description="Default arguments")

    def parseArguments(self, arguments, description=""):
        parser = argparse.ArgumentParser(description=description)
        for arg in arguments:
            arg.addParserArguments(parser)
        return parser.parse_known_args()[0]

    def transformArguments(self, context, args):
        self.loadArguments(args)
        self.formatArguments(context, self.getArguments(exclude="outputs"))
        arguments = {
            k: {arg.key: arg.value for arg in v if arg.isSet}
            for k, v in self.arguments.items()
        }
        arguments.update(
            {arg.key: arg.value for arg in itertools.chain(*self.arguments.values())}
        )
        return Arguments(**arguments)

    def loadArguments(self, args):
        for arg in self.getArguments():
            arg.load(args)

    def _safe_list(self, params=None):
        return [params] if isinstance(params, str) else list(params)

    def getArguments(self, include=None, exclude=None):
        includes = set(
            self.arguments.keys() if not include else self._safe_list(include)
        )
        excludes = set([] if not exclude else self._safe_list(exclude))
        includes = includes - excludes
        argumentsLists = [self.arguments[c] for c in includes]
        return itertools.chain(*argumentsLists)

    def formatArguments(self, context, arguments):
        for arg in arguments:
            arg.format(context)

    def saveMutipleOutputs(self, context, outputs, results):
        if isinstance(results, tuple) or isinstance(results, list):
            for argument, result in zip(outputs, results):
                argument.save(context, Result(result))
        elif isinstance(results, dict):
            notFoundResults = [
                argument.key for argument in outputs if argument.key not in results
            ]
            if notFoundResults:
                raise Exception("Not found results: {}".format(notFoundResults))
            for argument in outputs:
                result = results.get(argument.key)
                argument.save(context, Result(result))
        else:
            raise Exception("Incorrect results: {}".format(results))

    def saveOneOutput(self, context, output, results):
        result = (
            Result(results[output.key])
            if isinstance(results, dict) and output.key in results
            else Result(results)
        )
        output.save(context, result)

    def saveOutputs(self, context, results):
        logger.info("Saving...")
        outputs = list(self.getArguments("outputs"))
        if len(outputs) > 1:
            self.saveMutipleOutputs(context, outputs, results)
        elif len(outputs) == 1:
            self.saveOneOutput(context, outputs[0], results)

    def addArgument(self, arg, argtype="args"):
        self.arguments[argtype].insert(0, arg)

    @classmethod
    def arg(cls, argument, argtype="args"):
        def _dec(funcOrComponent):
            funcOrComponent = (
                funcOrComponent
                if isinstance(funcOrComponent, cls)
                else cls(funcOrComponent)
            )
            funcOrComponent.addArgument(argument, argtype=argtype)
            return funcOrComponent

        return _dec

    @classmethod
    def input(cls, argument):
        return cls.arg(argument, argtype="inputs")

    @classmethod
    def output(cls, argument):
        return cls.arg(argument, argtype="outputs")

    @classmethod
    def param(cls, argument):
        return cls.arg(argument, argtype="params")

    @classmethod
    def column(cls, argument):
        return cls.arg(argument, argtype="columns")
