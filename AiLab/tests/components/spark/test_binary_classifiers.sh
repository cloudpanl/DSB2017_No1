#!/bin/bash
set -xe

. tests/test_functions.sh


run_data_split iris_binary_tmp

declare -a arr=(
    "SPGBTClassifier"
)

for i in "${arr[@]}"
do
   run_classifer $i
   run_predictor
   run_evaluator SPBinaryClassificationEvaluator
done
