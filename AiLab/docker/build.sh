#!/bin/bash

set -xe

docker build -t registry.cn-shanghai.aliyuncs.com/shuzhi/$1:$2-$3 . -f ./docker/$1/Dockerfile --build-arg PYTHON_VERSION=$2
docker tag registry.cn-shanghai.aliyuncs.com/shuzhi/$1:$2-$3 registry.cn-shanghai.aliyuncs.com/shuzhi/$1:latest
docker tag registry.cn-shanghai.aliyuncs.com/shuzhi/$1:$2-$3 registry.cn-shanghai.aliyuncs.com/shuzhi/$1:$2
docker push registry.cn-shanghai.aliyuncs.com/shuzhi/$1:$2-$3
docker push registry.cn-shanghai.aliyuncs.com/shuzhi/$1:$2
docker push registry.cn-shanghai.aliyuncs.com/shuzhi/$1:latest
