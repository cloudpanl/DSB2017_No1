# coding=utf-8
from __future__ import print_function


class Proxy(object):
    MAPPING = {}

    def __init__(self):
        self._backend = None

    def __getattr__(self, key):
        return getattr(self.backend, key)

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def backend(self):
        if self._backend is None:
            raise Exception("Storage error: storage isn't set.")
        return self._backend

    def setBackend(self, *args, **kwargs):
        backendType = kwargs.get("type")
        if not backendType:
            raise Exception("{} set error: backend type is required".format(self.name))
        BackendClass = self.MAPPING.get(backendType)
        if not BackendClass:
            raise Exception(
                "{} don't supported backend type: {}".format(self.name, backendType)
            )
        self._backend = BackendClass(*args, **kwargs)
        return self._backend
