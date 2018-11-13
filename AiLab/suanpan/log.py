# coding=utf-8
from __future__ import print_function

import datetime
import logging
import logging.handlers
import os
import time


class Formatter(logging.Formatter):

    converter = time.gmtime

    def __init__(
        self,
        fmt="%(asctime)s :: %(levelname)-10s :: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    ):
        super(Formatter, self).__init__(fmt=fmt, datefmt=datefmt)


class Logger(logging.Logger):

    formatter = Formatter()
    streamLogLevel = logging.INFO
    fileLogLevel = logging.DEBUG
    logFileMaxSize = 1024 * 1024 * 1024
    backupCount = 5
    logPath = "logs"
    logFile = None

    def __init__(self, name="suanpan"):
        super(Logger, self).__init__(name=name)
        self.streamHandler = logging.StreamHandler()
        self.streamHandler.setLevel(self.streamLogLevel)
        self.streamHandler.setFormatter(self.formatter)
        self.addHandler(self.streamHandler)

        if not os.path.exists(self.logPath):
            os.makedirs(self.logPath)
        self.logFile = "{}.log".format(name) if not self.logFile else self.logFile
        self.fileHandler = logging.handlers.RotatingFileHandler(
            os.path.join(self.logPath, self.logFile),
            maxBytes=self.logFileMaxSize,
            backupCount=self.backupCount,
        )
        self.fileHandler.setLevel(self.fileLogLevel)
        self.fileHandler.setFormatter(self.formatter)
        self.addHandler(self.fileHandler)


class LoggerProxy(object):
    def __init__(self, loggerOrName):
        self.setLogger(loggerOrName)

    def __getattr__(self, key):
        return getattr(self.logger, key)

    def setLogger(self, loggerOrName):
        self.logger = (
            loggerOrName if isinstance(loggerOrName, Logger) else Logger(loggerOrName)
        )


rootLogger = Logger("suanpan")
logger = LoggerProxy(rootLogger)
