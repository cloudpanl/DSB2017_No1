#!/bin/bash
set -xe

pipenv run \
python components/docker/DSB2017/SPPreprocess.py  \
--hive-host "47.94.82.175" \
--hive-port "10000" \
--hive-username "spark" \
--storage-type 'oss' \
--storage-oss-access-id 'LTAIgV6cMz4TgHZB' \
--storage-oss-access-key 'M6jP8a1KN2kfZR51M08UiEufnzQuiY' \
--storage-oss-bucket-name 'suanpan' \
--storage-oss-temp-store 'tmp' \
--inputDataFolder 'majik_test/stage1_samples_2' \
--inputLabelCsv 'majik_test/stage1_labels' \
--outputNpy 'majik_test/SPPreprocess/outputNpy' \
--outputImages 'majik_test/SPPreprocess/outputImages'


pipenv run \
python components/docker/DSB2017/SPTrainKeras.py  \
--hive-host "47.94.82.175" \
--hive-port "10000" \
--hive-username "spark" \
--storage-type 'oss' \
--storage-oss-access-id 'LTAIgV6cMz4TgHZB' \
--storage-oss-access-key 'M6jP8a1KN2kfZR51M08UiEufnzQuiY' \
--storage-oss-bucket-name 'suanpan' \
--storage-oss-temp-store 'tmp' \
--inputDataNpy 'majik_test/SPPreprocess/outputNpy' \
--outputH5Model 'majik_test/SPTrainKeras/outputH5Model' \
--epochs 5


pipenv run \
python components/docker/DSB2017/SPPredictKeras.py  \
--hive-host "47.94.82.175" \
--hive-port "10000" \
--hive-username "spark" \
--storage-type 'oss' \
--storage-oss-access-id 'LTAIgV6cMz4TgHZB' \
--storage-oss-access-key 'M6jP8a1KN2kfZR51M08UiEufnzQuiY' \
--storage-oss-bucket-name 'suanpan' \
--storage-oss-temp-store 'tmp' \
--inputDataNpy 'majik_test/SPPreprocess/outputNpy' \
--inputH5Model 'majik_test/SPTrainKeras/outputH5Model' \
--outputTable 'SPPredictKerasTestTable'
