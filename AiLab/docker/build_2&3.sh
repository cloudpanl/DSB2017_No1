#!/bin/bash

set -xe

bash docker/build.sh $1 2 $2
bash docker/build.sh $1 3 $2
