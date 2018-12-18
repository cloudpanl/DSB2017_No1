#!/bin/bash

set -xe

IMAGE_NAME=horovod
IMAGE=registry.cn-shanghai.aliyuncs.com/shuzhi/${IMAGE_NAME}

docker build -t ${IMAGE}:$1 . -f ./docker/Horovod.Dockerfile ${@:2}

docker tag ${IMAGE}:$1 ${IMAGE}:latest

docker push ${IMAGE}:$1
docker push ${IMAGE}:latest
