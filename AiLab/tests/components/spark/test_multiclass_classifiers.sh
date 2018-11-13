#!/bin/bash
set -xe

. tests/test_functions.sh


run_data_split iris_tmp

declare -a arr=(
    "SPDecisionTreeClassifier"
    "SPLogisticRegression"
    "SPMultilayerPerceptronClassifier --layers 4,4,3,3"
    "SPNaiveBayes"
    "SPRandomForestClassifier"
)

for i in "${arr[@]}"
do
   run_classifer $i
   run_predictor
   run_evaluator SPMulticlassClassificationEvaluator
done
