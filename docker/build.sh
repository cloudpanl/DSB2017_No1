#!/bin/bash

IMAGE_NAME=dsb3_no1

docker build -t registry.cn-shanghai.aliyuncs.com/shuzhi/$IMAGE_NAME:$1 . -f ./docker/Dockerfile
docker tag registry.cn-shanghai.aliyuncs.com/shuzhi/$IMAGE_NAME:$1 registry.cn-shanghai.aliyuncs.com/shuzhi/$IMAGE_NAME:latest
docker push registry.cn-shanghai.aliyuncs.com/shuzhi/$IMAGE_NAME:$1
docker push registry.cn-shanghai.aliyuncs.com/shuzhi/$IMAGE_NAME:latest
