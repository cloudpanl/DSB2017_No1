set -xe

HIVE_HOST="1.1.1.1"
HIVE_PORT=10000

OSS_ACCESS_ID="********"
OSS_ACCESS_KEY="********"
OSS_BUCKET="********"

python component_dsb3_preprocess.py \
--dw-type "hive" \
--dw-hive-host ${HIVE_HOST} \
--dw-hive-port ${HIVE_PORT} \
--dw-hive-username "spark" \
--storage-type 'oss' \
--storage-oss-access-id ${OSS_ACCESS_ID} \
--storage-oss-access-key ${OSS_ACCESS_KEY} \
--storage-oss-bucket-name ${OSS_BUCKET} \
--storage-oss-temp-store 'tmp' \
--inputDataFolder "majik_test/DSB2017_Data/DSB3/stage1_samples_2" \
--inputLabelsFolder "majik_test/DSB2017_Data/DSB3/stage1_annos" \
--outputDataTable "component_dsb3_preprocess_output_table" \
--outputDataFolder "majik_test/component_dsb3_preprocess_output_data"

python component_luna_preprocess.py \
--dw-type "hive" \
--dw-hive-host ${HIVE_HOST} \
--dw-hive-port ${HIVE_PORT} \
--dw-hive-username "spark" \
--storage-type 'oss' \
--storage-oss-access-id ${OSS_ACCESS_ID} \
--storage-oss-access-key ${OSS_ACCESS_KEY} \
--storage-oss-bucket-name ${OSS_BUCKET} \
--storage-oss-temp-store 'tmp' \
--inputRawFolder "majik_test/DSB2017_Data/LUNA2016" \
--inputSegmentFolder "majik_test/DSB2017_Data/LUNA2016/seg-lungs-LUNA16" \
--inputAbbr "majik_test/DSB2017_Data/LUNA2016/labels/shorter" \
--inputLabels "majik_test/DSB2017_Data/LUNA2016/labels/lunaqualified" \
--outputDataTable "component_luna_preprocess_output_data" \
--outputDataFolder "majik_test/component_luna_preprocess_output_data"

python component_folder_combine.py \
--dw-type "hive" \
--dw-hive-host ${HIVE_HOST} \
--dw-hive-port ${HIVE_PORT} \
--dw-hive-username "spark" \
--storage-type 'oss' \
--storage-oss-access-id ${OSS_ACCESS_ID} \
--storage-oss-access-key ${OSS_ACCESS_KEY} \
--storage-oss-bucket-name ${OSS_BUCKET} \
--storage-oss-temp-store 'tmp' \
--inputFolder1 "majik_test/component_dsb3_preprocess_output_data" \
--inputFolder2 "majik_test/component_luna_preprocess_output_data" \
--outputFolder "majik_test/component_folder_combine_output_data"


docker run \
--runtime=nvidia \
--network=host \
--ipc=host \
-e CUDA_VISIBLE_DEVICES=1,2 \
-v kaggle:/sp_data \
registry.cn-shanghai.aliyuncs.com/shuzhi/kaggle_no1_component:GPU \
mpirun -np 2 -H localhost:2 \
python component_n_net_train.py \
--dw-type "hive" \
--dw-hive-host "47.94.82.175" \
--dw-hive-port "10000" \
--dw-hive-username "spark" \
--storage-type 'oss' \
--storage-oss-access-id 'LTAIgV6cMz4TgHZB' \
--storage-oss-access-key 'M6jP8a1KN2kfZR51M08UiEufnzQuiY' \
--storage-oss-bucket-name 'suanpan' \
--storage-oss-temp-store '/sp_data' \
--inputTrainDataTable "AIStudio_temp_648_9b73f710edfc11e89b0b8b30232e5b99_out1" \
--inputValidateDataTable "AIStudio_temp_648_9b73f710edfc11e89b0b8b30232e5b99_out2" \
--inputDataFolder "studio/shanglu/648/b78817f0efec11e88cdf854bf5061962/out1/data" \
--outputCheckpoint "majik_test/component_n_net_train_output_checkpoint" \
--idColumn "patient" \
--batchSize "4"

-e CUDA_VISIBLE_DEVICES=1,2 \

docker run \
--runtime=nvidia \
--network=host \
--ipc=host \
-v kaggle:/sp_data \
registry.cn-shanghai.aliyuncs.com/shuzhi/kaggle_no1_component:GPU \
mpirun -np 2 -H 172.26.152.220:1,172.26.152.219:1 \
-mca plm_rsh_args "-p 8880" \
python component_n_net_train.py \
--dw-type "hive" \
--dw-hive-host "47.94.82.175" \
--dw-hive-port "10000" \
--dw-hive-username "spark" \
--storage-type 'oss' \
--storage-oss-access-id 'LTAIgV6cMz4TgHZB' \
--storage-oss-access-key 'M6jP8a1KN2kfZR51M08UiEufnzQuiY' \
--storage-oss-bucket-name 'suanpan' \
--storage-oss-temp-store '/sp_data' \
--inputTrainDataTable "AIStudio_temp_648_9b73f710edfc11e89b0b8b30232e5b99_out1" \
--inputValidateDataTable "AIStudio_temp_648_9b73f710edfc11e89b0b8b30232e5b99_out2" \
--inputDataFolder "studio/shanglu/648/b78817f0efec11e88cdf854bf5061962/out1/data" \
--outputCheckpoint "majik_test/component_n_net_train_output_checkpoint" \
--idColumn "patient" \
--batchSize "4"

docker run \
--runtime=nvidia \
--network=host \
--ipc=host \
-v kaggle:/sp_data \
registry.cn-shanghai.aliyuncs.com/shuzhi/kaggle_no1_component:GPU \
bash -c "/usr/sbin/sshd -p 8880; sleep infinity"


docker run --rm registry.cn-shanghai.aliyuncs.com/shuzhi/dsb3_no1 \
python component_n_net_predict.py \
--dw-type "hive" \
--dw-hive-host ${HIVE_HOST} \
--dw-hive-port ${HIVE_PORT} \
--dw-hive-username "spark" \
--storage-type 'oss' \
--storage-oss-access-id ${OSS_ACCESS_ID} \
--storage-oss-access-key ${OSS_ACCESS_KEY} \
--storage-oss-bucket-name ${OSS_BUCKET} \
--storage-oss-temp-store 'tmp' \
--inputDataTable "component_dsb3_preprocess_output_table" \
--inputDataFolder "majik_test/component_folder_combine_output_data" \
--inputCheckpoint "majik_test/component_n_net_predict_input_checkpoint/detector" \
--outputBboxDataTable "component_n_net_predict_output_bbox_data" \
--outputBboxFolder "majik_test/component_n_net_predict_output_bbox_data" \
--idColumn "patient"

python component_data_to_images.py \
--dw-type "hive" \
--dw-hive-host ${HIVE_HOST} \
--dw-hive-port ${HIVE_PORT} \
--dw-hive-username "spark" \
--storage-type 'oss' \
--storage-oss-access-id ${OSS_ACCESS_ID} \
--storage-oss-access-key ${OSS_ACCESS_KEY} \
--storage-oss-bucket-name ${OSS_BUCKET} \
--storage-oss-temp-store 'tmp' \
--inputDataTable "component_dsb3_preprocess_output_table" \
--inputDataFolder "majik_test/component_folder_combine_output_data" \
--outputImagesFolder "majik_test/component_data_to_image_output_images" \
--idColumn "patient" \
--dataColumn "image_path"

python component_data_to_mask_images.py \
--dw-type "hive" \
--dw-hive-host ${HIVE_HOST} \
--dw-hive-port ${HIVE_PORT} \
--dw-hive-username "spark" \
--storage-type 'oss' \
--storage-oss-access-id ${OSS_ACCESS_ID} \
--storage-oss-access-key ${OSS_ACCESS_KEY} \
--storage-oss-bucket-name ${OSS_BUCKET} \
--storage-oss-temp-store 'tmp' \
--inputDataTable "component_n_net_predict_output_bbox_data" \
--inputDataFolder "majik_test/component_folder_combine_output_data" \
--inputBboxDataFolder "majik_test/component_n_net_predict_output_bbox_data" \
--outputImagesFolder "majik_test/component_data_to_mask_images_output_images" \
--idColumn "patient" \
--dataColumn "image_path"

# Predict

python component_predict_preprocess.py \
--dw-type "hive" \
--dw-hive-host ${HIVE_HOST} \
--dw-hive-port ${HIVE_PORT} \
--dw-hive-username "spark" \
--storage-type 'oss' \
--storage-oss-access-id ${OSS_ACCESS_ID} \
--storage-oss-access-key ${OSS_ACCESS_KEY} \
--storage-oss-bucket-name ${OSS_BUCKET} \
--storage-oss-temp-store 'tmp' \
--inputDataFolder "majik_test/DSB2017_Data/DSB3/stage1_samples" \
--outputDataTable "component_predict_preprocess_output_data" \
--outputDataFolder "majik_test/component_predict_preprocess_output_data"

python component_n_net_predict.py \
--dw-type "hive" \
--dw-hive-host ${HIVE_HOST} \
--dw-hive-port ${HIVE_PORT} \
--dw-hive-username "spark" \
--storage-type 'oss' \
--storage-oss-access-id ${OSS_ACCESS_ID} \
--storage-oss-access-key ${OSS_ACCESS_KEY} \
--storage-oss-bucket-name ${OSS_BUCKET} \
--storage-oss-temp-store 'tmp' \
--inputDataTable "component_predict_preprocess_output_data" \
--inputDataFolder "majik_test/component_predict_preprocess_output_data" \
--inputCheckpoint "majik_test/component_n_net_predict_input_checkpoint/detector" \
--outputBboxDataTable "component_n_net_predict_output_bbox_data_predict" \
--outputBboxFolder "majik_test/component_n_net_predict_output_bbox_data_predict" \
--idColumn "patient"

python component_data_to_mask_images.py \
--dw-type "hive" \
--dw-hive-host ${HIVE_HOST} \
--dw-hive-port ${HIVE_PORT} \
--dw-hive-username "spark" \
--storage-type 'oss' \
--storage-oss-access-id ${OSS_ACCESS_ID} \
--storage-oss-access-key ${OSS_ACCESS_KEY} \
--storage-oss-bucket-name ${OSS_BUCKET} \
--storage-oss-temp-store 'tmp' \
--inputDataTable "component_n_net_predict_output_bbox_data_predict" \
--inputDataFolder "majik_test/component_predict_preprocess_output_data" \
--inputBboxDataFolder "majik_test/component_n_net_predict_output_bbox_data_predict" \
--outputImagesFolder "majik_test/component_data_to_mask_images_output_images_predict" \
--idColumn "patient" \
--dataColumn "image_path"

###############################

docker run -d -v kaggle:/sp_data registry.cn-shanghai.aliyuncs.com/shuzhi/kaggle_no1_component:CPU \
python /home/DSB3/component_dsb3_preprocess.py \
--inputDataFolder 'studio/shanglu/majik_test/wly/data' \
--inputLabelsFolder 'studio/shanglu/majik_test/wly/annos' \
--outputDataTable 'wly_preprocess_data' \
--outputDataFolder 'studio/shanglu/majik_test/wly/preprocess' \
--storage-type 'oss' \
--storage-oss-endpoint 'http://oss-cn-beijing.aliyuncs.com' \
--storage-oss-bucket-name 'suanpan' \
--storage-oss-access-id 'LTAIgV6cMz4TgHZB' \
--storage-oss-access-key 'M6jP8a1KN2kfZR51M08UiEufnzQuiY' \
--storage-oss-temp-store '/sp_data' \
--dw-type 'hive' \
--dw-hive-host '47.94.82.175' \
--dw-hive-port '10000' \
--dw-hive-database 'default' \
--dw-hive-auth '' \
--dw-hive-username '' \
--dw-hive-password ''

docker run \
-d \
--runtime=nvidia \
--network=host \
--ipc=host \
-v kaggle:/sp_data \
registry.cn-shanghai.aliyuncs.com/shuzhi/kaggle_no1_component:GPU \
/usr/sbin/sshd -p 12345 -D

docker run \
-d \
--network=host \
--runtime=nvidia \
--ipc=host \
-v kaggle:/sp_data \
registry.cn-shanghai.aliyuncs.com/shuzhi/kaggle_no1_component:GPU \
mpirun -np 5 -H 172.26.152.224:1,172.26.152.225:1,172.26.152.223:1,172.26.152.220:1,172.26.152.219:1 \
-mca plm_rsh_args "-p 12345" \
python component_n_net_train.py \
--storage-type 'oss' \
--storage-oss-endpoint 'http://oss-cn-beijing.aliyuncs.com' \
--storage-oss-bucket-name 'suanpan' \
--storage-oss-access-id 'LTAIgV6cMz4TgHZB' \
--storage-oss-access-key 'M6jP8a1KN2kfZR51M08UiEufnzQuiY' \
--storage-oss-temp-store '/sp_data' \
--dw-type 'hive' \
--dw-hive-host '47.94.82.175' \
--dw-hive-port '10000' \
--dw-hive-database 'default' \
--dw-hive-auth '' \
--dw-hive-username '' \
--dw-hive-password '' \
--inputTrainDataTable "AIStudio_temp_648_c80edda0131111e9a9075b5b430fcf9b_out1" \
--inputValidateDataTable "AIStudio_temp_648_c80edda0131111e9a9075b5b430fcf9b_out2" \
--inputDataFolder "/sp_data/studio/shanglu/648/5c590360131111e9a9075b5b430fcf9b/out2/data" \
--outputCheckpoint "studio/shanglu/majik_test/wly/component_n_net_train_output_checkpoint" \
--idColumn "patient" \
--batchSize "8"
