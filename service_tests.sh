set -xe

HIVE_HOST="1.1.1.1"
HIVE_PORT=10000

OSS_ACCESS_ID="********"
OSS_ACCESS_KEY="********"
OSS_BUCKET="********"

pipenv run \
python service_preprocess.py \
--dw-type "hive" \
--dw-hive-host ${HIVE_HOST} \
--dw-hive-port ${HIVE_PORT} \
--dw-hive-username "spark" \
--storage-type 'oss' \
--storage-oss-access-id ${OSS_ACCESS_ID} \
--storage-oss-access-key ${OSS_ACCESS_KEY} \
--storage-oss-bucket-name ${OSS_BUCKET} \
--storage-oss-temp-store 'tmp' \
--port 8981

pipenv run \
python service_predict.py \
--dw-type "hive" \
--dw-hive-host ${HIVE_HOST} \
--dw-hive-port ${HIVE_PORT} \
--dw-hive-username "spark" \
--storage-type 'oss' \
--storage-oss-access-id ${OSS_ACCESS_ID} \
--storage-oss-access-key ${OSS_ACCESS_KEY} \
--storage-oss-bucket-name ${OSS_BUCKET} \
--storage-oss-temp-store 'tmp' \
--port 8982

pipenv run \
python service_dector.py \
--dw-type "hive" \
--dw-hive-host ${HIVE_HOST} \
--dw-hive-port ${HIVE_PORT} \
--dw-hive-username "spark" \
--storage-type 'oss' \
--storage-oss-access-id ${OSS_ACCESS_ID} \
--storage-oss-access-key ${OSS_ACCESS_KEY} \
--storage-oss-bucket-name ${OSS_BUCKET} \
--storage-oss-temp-store 'tmp' \
--port 8983
