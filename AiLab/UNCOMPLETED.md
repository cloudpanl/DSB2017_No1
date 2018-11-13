# Uncompleted

## Classifation
- **spark.ml.classifation.SPLinearSVC**: PMML not support LinearSVCModel
- **spark.ml.classifation.SPOneVsRest**: OneVsRest is not standard classifier

## Regression
- **ALL**: PMML only support double schema for all regressors
- **spark.ml.regression.SPAFTSurvivalRegression**: The lifetime or label should be  greater than 0.0
- **spark.ml.regression.SPIsotonicRegression**: PMML not support IsotonicRegression

## Evalutor
- **spark.ml.classifation.SPClusteringEvaluator**: ClusteringEvaluator requires sparkml >= 2.3.0

## Feature
- **spark.ml.classifation.SPFeatureHasher**: FeatureHasher requires sparkml >= 2.3.0
- **spark.ml.classifation.SPOneHotEncoderEstimator**: OneHotEncoderEstimator requires sparkml >= 2.3.0
- **spark.ml.classifation.SPVectorSizeHint**: VectorSizeHint requires sparkml >= 2.3.0
