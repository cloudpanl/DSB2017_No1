#!/bin/bash

set -xe

bash docker/build_component.sh CPU $@
bash docker/build_component.sh GPU $@
bash docker/build_service.sh CPU $@
bash docker/build_service.sh GPU $@
