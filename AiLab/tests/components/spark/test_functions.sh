#!/bin/bash
set -xe

export PY2ENV=./py2env
export LIB_JAR=./lib.jar

run_data_split() {
    table=$1 && shift
    PYSPARK_PYTHON=${PY2ENV}/bin/python \
    spark-submit \
    --master local[*] \
    --jars ${LIB_JAR} \
    components/spark/data/SPRandomSpliter.py \
    --inputTable $table \
    --outputTrainTable 'majik_temp_train' \
    --outputTestTable 'majik_temp_test' \
    $*
}

run_classifer() {
    file=$1 && shift
    PYSPARK_PYTHON=${PY2ENV}/bin/python \
    spark-submit \
    --master local[*] \
    --jars ${LIB_JAR} \
    components/spark/ml/classification/$file.py \
    --inputTable 'majik_temp_train' \
    --outputModel 'majik_temp_model' \
    --labelColumn 'class' \
    --selectColumns 'sepal_length, sepal_width, petal_length, petal_width' \
    $*
}

run_regressor() {
    file=$1 && shift
    PYSPARK_PYTHON=${PY2ENV}/bin/python \
    spark-submit \
    --master local[*] \
    --jars ${LIB_JAR} \
    components/spark/ml/regression/$file.py \
    --inputTable 'majik_temp_train' \
    --outputModel 'majik_temp_model' \
    --labelColumn 'petal_width' \
    --selectColumns 'sepal_length, sepal_width, petal_length' \
    $*
}

run_feature() {
    file=$1 && shift
    PYSPARK_PYTHON=${PY2ENV}/bin/python \
    spark-submit \
    --master local[*] \
    --jars ${LIB_JAR} \
    components/spark/ml/feature/$file.py \
    $*
}

run_predictor() {
    PYSPARK_PYTHON=${PY2ENV}/bin/python \
    spark-submit \
    --master local[*] \
    --jars ${LIB_JAR} \
    components/spark/ml/SPPredictor.py \
    --inputModel 'majik_temp_model' \
    --inputTable 'majik_temp_test' \
    --outputTable 'majik_temp_prediction' \
    $*
}

run_evaluator() {
    file=$1 && shift
    PYSPARK_PYTHON=${PY2ENV}/bin/python \
    spark-submit \
    --master local[*] \
    --jars ${LIB_JAR} \
    components/spark/ml/evaluation/$file.py \
    --inputTable 'majik_temp_model' \
    --inputTable 'majik_temp_prediction' \
    --outputTable 'majik_temp_evaluation' \
    $*
}
