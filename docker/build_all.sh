#!/bin/bash

set -xe

bash docker/build_component.sh CPU $1
bash docker/build_component.sh GPU $1
bash docker/build_service.sh CPU $1
bash docker/build_service.sh GPU $1
