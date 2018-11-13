#!/bin/bash

set -xe

# docker login first of all

declare -a arr=(
    "conda_base 1.0"
    "py2env 0.3.4"
    "docker_component 1.6"
)

for i in "${arr[@]}"
do
   bash docker/build.sh $i
done
