#!/bin/bash

set -xe

docker pull registry.cn-shanghai.aliyuncs.com/shuzhi/pyenv:$1
docker run --rm -v ${PWD}:/dist registry.cn-shanghai.aliyuncs.com/shuzhi/pyenv:$1
