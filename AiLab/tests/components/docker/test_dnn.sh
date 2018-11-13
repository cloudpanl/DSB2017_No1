#!/bin/bash

set -xe

pipenv run \
python components/docker/tensorflow/SPDnnPreprocess.py \
--dw-type "hive" \
--dw-hive-host "47.94.82.175" \
--dw-hive-port "10000" \
--dw-hive-username "spark" \
--storage-type 'oss' \
--storage-oss-access-id 'LTAIgV6cMz4TgHZB' \
--storage-oss-access-key 'M6jP8a1KN2kfZR51M08UiEufnzQuiY' \
--storage-oss-bucket-name 'suanpan' \
--storage-oss-temp-store 'tmp' \
--outputTable "dnn_data_test" \
--outputDataFolder "majik_test/dnn_data_test"


pipenv run \
docker run --rm registry.cn-shanghai.aliyuncs.com/shuzhi/tensorflow_component:2-1.0 \
python components/docker/tensorflow/SPDnn.py \
--dw-type "hive" \
--dw-hive-host "47.94.82.175" \
--dw-hive-port "10000" \
--dw-hive-username "spark" \
--storage-type 'oss' \
--storage-oss-access-id 'LTAIgV6cMz4TgHZB' \
--storage-oss-access-key 'M6jP8a1KN2kfZR51M08UiEufnzQuiY' \
--storage-oss-bucket-name 'suanpan' \
--storage-oss-temp-store 'tmp' \
--outputModel "majik_test/dnn_model_test" \
--numClasses "10"
--inputShape "784"
--hiddenUnits "512,256"

pipenv run \
python components/docker/tensorflow/SPDnnTrain.py \
--dw-type "hive" \
--dw-hive-host "47.94.82.175" \
--dw-hive-port "10000" \
--dw-hive-username "spark" \
--storage-type 'oss' \
--storage-oss-access-id 'LTAIgV6cMz4TgHZB' \
--storage-oss-access-key 'M6jP8a1KN2kfZR51M08UiEufnzQuiY' \
--storage-oss-bucket-name 'suanpan' \
--storage-oss-temp-store 'tmp' \
--outputTable "dnn_data_test" \
--outputDataFolder "majik_test/dnn_data_test"
