set -xe

pipenv run \
python service_preprocess.py \
--storage-type 'oss' \
--storage-oss-access-id 'LTAIgV6cMz4TgHZB' \
--storage-oss-access-key 'M6jP8a1KN2kfZR51M08UiEufnzQuiY' \
--storage-oss-bucket-name 'suanpan' \
--storage-oss-temp-store 'tmp' \
--port 8981


pipenv run \
python service_predict.py \
--storage-type 'oss' \
--storage-oss-access-id 'LTAIgV6cMz4TgHZB' \
--storage-oss-access-key 'M6jP8a1KN2kfZR51M08UiEufnzQuiY' \
--storage-oss-bucket-name 'suanpan' \
--storage-oss-temp-store 'tmp' \
--port 8982


pipenv run \
python service_dector.py \
--storage-type 'oss' \
--storage-oss-access-id 'LTAIgV6cMz4TgHZB' \
--storage-oss-access-key 'M6jP8a1KN2kfZR51M08UiEufnzQuiY' \
--storage-oss-bucket-name 'suanpan' \
--storage-oss-temp-store 'tmp' \
--port 8983
