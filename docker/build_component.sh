#!/bin/bash

set -xe

IMAGE_NAME=kaggle_no1_component
IMAGE=registry.cn-shanghai.aliyuncs.com/shuzhi/${IMAGE_NAME}

docker build -t ${IMAGE}:$1-$2 . -f ./docker/Component_$1.Dockerfile ${@:3}

docker tag ${IMAGE}:$1-$2 ${IMAGE}:$1
docker tag ${IMAGE}:$1-$2 ${IMAGE}:latest

docker push ${IMAGE}:$1-$2
docker push ${IMAGE}:$1
docker push ${IMAGE}:latest
