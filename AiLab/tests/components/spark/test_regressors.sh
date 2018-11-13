#!/bin/bash
set -xe

. tests/test_functions.sh


run_data_split iris_tmp

declare -a arr=(
    "SPDecisionTreeRegressor"
    "SPGBTRegressor"
    "SPGeneralizedLinearRegression --link identity"
    "SPLinearRegression"
    "SPRandomForestRegressor"
)

for i in "${arr[@]}"
do
   run_regressor $i
   run_predictor
   run_evaluator SPRegressionEvaluator
done
