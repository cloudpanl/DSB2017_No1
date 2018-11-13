#!/bin/bash
set -xe

pipenv run pip install .

pipenv run \
python components/docker/SPDateTimeCombine.py \
--hive-host "47.94.82.175" \
--hive-port "10000" \
--hive-username "spark" \
--inputTable "working_parameter_crec188_mi_preview_test" \
--outputTable "working_parameter_crec188_mi_preview_test2" \
--dataTimeColumn "datetime" \
--dataTimeStringColumn "datetime_string" \
--selectedColumns "year,month,day,hour,minute"

pipenv run \
python components/docker/SPWaveExtract.py \
--hive-host "47.94.82.175" \
--hive-port "10000" \
--hive-username "spark" \
--inputTable "working_parameter_crec188_mi_preview_test2" \
--outputTable "working_parameter_crec188_mi_preview_test3" \
--periodColumn "period" \
--selectedColumns "speed_avg,speed1_avg,num1_current,force_avg,cutter_head_rev_avg,num1_torque_avg,num1_out_power_avg,cutter_head_power_avg" \
--condition "all([data[c] > 0 for c in columns])"

pipenv run \
python components/docker/SPForwardReverseSearch.py \
--hive-host "47.94.82.175" \
--hive-port "10000" \
--hive-username "spark" \
--inputTable "working_parameter_crec188_mi_preview_test3" \
--outputTable "working_parameter_crec188_mi_preview_test4" \
--periodColumn "period" \
--stageColumn "stage" \
--selectedColumns "speed_avg,speed1_avg,num1_current,force_avg,cutter_head_rev_avg,num1_torque_avg,num1_out_power_avg,cutter_head_power_avg" \
--forwardCondition 'any([data[c] > mean[c] - std[c] / 2 for c in columns])' \
--reverseCondition 'any([data[c] > mean[c] - std[c] / 2 for c in columns])'
