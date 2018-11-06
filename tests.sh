set -xe

python component_inference_1_preprocess.py \
--inputDataFolder "../DSB2017_Data/DSB3/stage1_samples" \
--outputDataFolder "output_data" \
--outputImagesFolder "output_images" \
--outputData "data/data_1.csv"

python component_inference_2_n_net.py \
--inputData "data/data_1.csv" \
--inputDataFolder "output_data" \
--inputModel "dsb/model/detector.ckpt" \
--outputBboxData "data/data_2.csv" \
--outputBboxFolder "output_bbox" \
--idColumn "patient"

python component_inference_3_mask.py \
--inputData "data/data_2.csv" \
--inputDataFolder "output_data" \
--inputBboxDataFolder "output_bbox" \
--outputImageFolder "output_nodules" \
--idColumn "patient"

python component_train_1_preprocess.py \
--dw-type "hive" \
--dw-hive-host "47.94.82.175" \
--dw-hive-port "10000" \
--dw-hive-username "spark" \
--storage-type 'oss' \
--storage-oss-access-id 'LTAIgV6cMz4TgHZB' \
--storage-oss-access-key 'M6jP8a1KN2kfZR51M08UiEufnzQuiY' \
--storage-oss-bucket-name 'suanpan' \
--storage-oss-temp-store 'tmp' \
--inputStage1Folder '../DSB2017_Data/DSB3/stage1_samples_2' \
--inputLunaRawFolder '../DSB2017_Data/LUNA2016' \
--inputLunaSegmentFolder '../DSB2017_Data/seg-lungs-LUNA16'
