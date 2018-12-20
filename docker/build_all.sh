#!/bin/bash

set -xe

VERSION="2.6"

bash docker/build_component.sh CPU ${VERSION} $@
bash docker/build_component.sh GPU ${VERSION} $@
bash docker/build_service.sh CPU ${VERSION} $@
bash docker/build_service.sh GPU ${VERSION} $@
