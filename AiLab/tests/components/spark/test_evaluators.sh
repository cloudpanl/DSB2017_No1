#!/bin/bash
set -xe

. tests/test_functions.sh


run_data_split iris_tmp
run_classifer SPDecisionTreeClassifier
run_predictor

declare -a arr=(
    "SPBinaryClassificationEvaluator"
    "SPMulticlassClassificationEvaluator"
    "SPRegressionEvaluator"
)

for i in "${arr[@]}"
do
   run_evaluator $i
done
