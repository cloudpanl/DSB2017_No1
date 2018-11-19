set -xe

pipenv run \
python component_dsb3_preprocess.py \
--dw-type "hive" \
--dw-hive-host "47.94.82.175" \
--dw-hive-port "10000" \
--dw-hive-username "spark" \
--storage-type 'oss' \
--storage-oss-access-id 'LTAIgV6cMz4TgHZB' \
--storage-oss-access-key 'M6jP8a1KN2kfZR51M08UiEufnzQuiY' \
--storage-oss-bucket-name 'suanpan' \
--storage-oss-temp-store 'tmp' \
--inputDataFolder "majik_test/DSB2017_Data/DSB3/stage1_samples_2" \
--inputLabelsFolder "majik_test/DSB2017_Data/DSB3/stage1_annos" \
--outputDataTable "component_dsb3_preprocess_output_table" \
--outputDataFolder "majik_test/component_dsb3_preprocess_output_data"

pipenv run \
python component_luna_preprocess.py \
--dw-type "hive" \
--dw-hive-host "47.94.82.175" \
--dw-hive-port "10000" \
--dw-hive-username "spark" \
--storage-type 'oss' \
--storage-oss-access-id 'LTAIgV6cMz4TgHZB' \
--storage-oss-access-key 'M6jP8a1KN2kfZR51M08UiEufnzQuiY' \
--storage-oss-bucket-name 'suanpan' \
--storage-oss-temp-store 'tmp' \
--inputRawFolder "majik_test/DSB2017_Data/LUNA2016" \
--inputSegmentFolder "majik_test/DSB2017_Data/LUNA2016/seg-lungs-LUNA16" \
--inputAbbr "majik_test/DSB2017_Data/LUNA2016/labels/shorter" \
--inputLabels "majik_test/DSB2017_Data/LUNA2016/labels/lunaqualified" \
--outputDataTable "component_luna_preprocess_output_data" \
--outputDataFolder "majik_test/component_luna_preprocess_output_data"

pipenv run \
python component_folder_combine.py \
--dw-type "hive" \
--dw-hive-host "47.94.82.175" \
--dw-hive-port "10000" \
--dw-hive-username "spark" \
--storage-type 'oss' \
--storage-oss-access-id 'LTAIgV6cMz4TgHZB' \
--storage-oss-access-key 'M6jP8a1KN2kfZR51M08UiEufnzQuiY' \
--storage-oss-bucket-name 'suanpan' \
--storage-oss-temp-store 'tmp' \
--inputFolder1 "majik_test/component_dsb3_preprocess_output_data" \
--inputFolder2 "majik_test/component_luna_preprocess_output_data" \
--outputFolder "majik_test/component_folder_combine_output_data"

pipenv run \
python component_n_net_train.py \
--dw-type "hive" \
--dw-hive-host "47.94.82.175" \
--dw-hive-port "10000" \
--dw-hive-username "spark" \
--storage-type 'oss' \
--storage-oss-access-id 'LTAIgV6cMz4TgHZB' \
--storage-oss-access-key 'M6jP8a1KN2kfZR51M08UiEufnzQuiY' \
--storage-oss-bucket-name 'suanpan' \
--storage-oss-temp-store 'tmp' \
--inputTrainDataTable "component_dsb3_preprocess_output_table" \
--inputValidateDataTable "component_luna_preprocess_output_data" \
--inputDataFolder "majik_test/component_folder_combine_output_data" \
--outputCheckpoint "majik_test/component_n_net_train_output_checkpoint" \
--idColumn "patient"

pipenv run \
docker run --rm registry.cn-shanghai.aliyuncs.com/shuzhi/dsb3_no1 \
python component_n_net_predict.py \
--dw-type "hive" \
--dw-hive-host "47.94.82.175" \
--dw-hive-port "10000" \
--dw-hive-username "spark" \
--storage-type 'oss' \
--storage-oss-access-id 'LTAIgV6cMz4TgHZB' \
--storage-oss-access-key 'M6jP8a1KN2kfZR51M08UiEufnzQuiY' \
--storage-oss-bucket-name 'suanpan' \
--storage-oss-temp-store 'tmp' \
--inputDataTable "component_dsb3_preprocess_output_table" \
--inputDataFolder "majik_test/component_folder_combine_output_data" \
--inputCheckpoint "majik_test/component_n_net_predict_input_checkpoint/detector" \
--outputBboxDataTable "component_n_net_predict_output_bbox_data" \
--outputBboxFolder "majik_test/component_n_net_predict_output_bbox_data" \
--idColumn "patient"

pipenv run \
python component_data_to_images.py \
--dw-type "hive" \
--dw-hive-host "47.94.82.175" \
--dw-hive-port "10000" \
--dw-hive-username "spark" \
--storage-type 'oss' \
--storage-oss-access-id 'LTAIgV6cMz4TgHZB' \
--storage-oss-access-key 'M6jP8a1KN2kfZR51M08UiEufnzQuiY' \
--storage-oss-bucket-name 'suanpan' \
--storage-oss-temp-store 'tmp' \
--inputDataTable "component_dsb3_preprocess_output_table" \
--inputDataFolder "majik_test/component_folder_combine_output_data" \
--outputImagesFolder "majik_test/component_data_to_image_output_images" \
--idColumn "patient" \
--dataColumn "image_path"

pipenv run \
python component_data_to_mask_images.py \
--dw-type "hive" \
--dw-hive-host "47.94.82.175" \
--dw-hive-port "10000" \
--dw-hive-username "spark" \
--storage-type 'oss' \
--storage-oss-access-id 'LTAIgV6cMz4TgHZB' \
--storage-oss-access-key 'M6jP8a1KN2kfZR51M08UiEufnzQuiY' \
--storage-oss-bucket-name 'suanpan' \
--storage-oss-temp-store 'tmp' \
--inputDataTable "component_n_net_predict_output_bbox_data" \
--inputDataFolder "majik_test/component_folder_combine_output_data" \
--inputBboxDataFolder "majik_test/component_n_net_predict_output_bbox_data" \
--outputImagesFolder "majik_test/component_data_to_mask_images_output_images" \
--idColumn "patient" \
--dataColumn "image_path"

# Predict

pipenv run \
python component_predict_preprocess.py \
--dw-type "hive" \
--dw-hive-host "47.94.82.175" \
--dw-hive-port "10000" \
--dw-hive-username "spark" \
--storage-type 'oss' \
--storage-oss-access-id 'LTAIgV6cMz4TgHZB' \
--storage-oss-access-key 'M6jP8a1KN2kfZR51M08UiEufnzQuiY' \
--storage-oss-bucket-name 'suanpan' \
--storage-oss-temp-store 'tmp' \
--inputDataFolder "majik_test/DSB2017_Data/DSB3/stage1_samples" \
--outputDataTable "component_predict_preprocess_output_data" \
--outputDataFolder "majik_test/component_predict_preprocess_output_data"

pipenv run \
python component_n_net_predict.py \
--dw-type "hive" \
--dw-hive-host "47.94.82.175" \
--dw-hive-port "10000" \
--dw-hive-username "spark" \
--storage-type 'oss' \
--storage-oss-access-id 'LTAIgV6cMz4TgHZB' \
--storage-oss-access-key 'M6jP8a1KN2kfZR51M08UiEufnzQuiY' \
--storage-oss-bucket-name 'suanpan' \
--storage-oss-temp-store 'tmp' \
--inputDataTable "component_predict_preprocess_output_data" \
--inputDataFolder "majik_test/component_predict_preprocess_output_data" \
--inputCheckpoint "majik_test/component_n_net_predict_input_checkpoint/detector" \
--outputBboxDataTable "component_n_net_predict_output_bbox_data_predict" \
--outputBboxFolder "majik_test/component_n_net_predict_output_bbox_data_predict" \
--idColumn "patient"

pipenv run \
python component_data_to_mask_images.py \
--dw-type "hive" \
--dw-hive-host "47.94.82.175" \
--dw-hive-port "10000" \
--dw-hive-username "spark" \
--storage-type 'oss' \
--storage-oss-access-id 'LTAIgV6cMz4TgHZB' \
--storage-oss-access-key 'M6jP8a1KN2kfZR51M08UiEufnzQuiY' \
--storage-oss-bucket-name 'suanpan' \
--storage-oss-temp-store 'tmp' \
--inputDataTable "component_n_net_predict_output_bbox_data_predict" \
--inputDataFolder "majik_test/component_predict_preprocess_output_data" \
--inputBboxDataFolder "majik_test/component_n_net_predict_output_bbox_data_predict" \
--outputImagesFolder "majik_test/component_data_to_mask_images_output_images_predict" \
--idColumn "patient" \
--dataColumn "image_path"