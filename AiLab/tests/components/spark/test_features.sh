#!/bin/bash
set -xe

. tests/test_functions.sh


run_feature SPRFormula \
--inputTable 'iris_tmp' \
--outputTable 'iris_rformula_tmp' \
--selectColumns 'sepal_length,sepal_width,petal_length,petal_width' \
--labelColumn 'class'


run_feature SPVectorAssembler \
--inputTable 'iris_tmp' \
--outputTable 'majik_feature_vectorassembler_temp' \
--inputCols 'sepal_length,sepal_width,petal_length,petal_width' \
--outputCol 'features'


run_feature SPBinarizer \
--inputTable 'iris_rformula_tmp' \
--outputTable 'majik_feature_binarizer_temp' \
--inputCol 'features' \
--outputCol 'binarizer_features'


run_feature SPBucketedRandomProjectionLSH \
--inputTable 'iris_rformula_tmp' \
--outputTable 'majik_feature_bucketedrandomprojectionlsh_temp' \
--inputCol 'features' \
--outputCol 'bucketedrandomprojectionlsh_features' \
--bucketLength '1.0'


run_feature SPBucketizer \
--inputTable 'iris_rformula_tmp' \
--outputTable 'majik_feature_bucketizer_temp' \
--inputCol 'sepal_length' \
--outputCol 'bucketizer_sepal_length' \
--splits '0,0.5,1.4,inf'


run_feature SPChiSqSelector \
--inputTable 'iris_rformula_tmp' \
--outputTable 'majik_feature_bucketizer_temp' \
--outputCol 'selected_features'


run_feature SPCountVectorizer \
--inputTable 'iris_rformula_tmp' \
--outputTable 'majik_feature_countvectorizer_temp' \
--inputCol 'values' \
--outputCol 'vectors'


run_feature SPDCT \
--inputTable 'iris_rformula_tmp' \
--outputTable 'majik_feature_dct_temp' \
--inputCol 'features' \
--outputCol 'vec'


run_feature SPElementwiseProduct \
--inputTable 'iris_rformula_tmp' \
--outputTable 'majik_feature_elementwiseproduct_temp' \
--inputCol 'features' \
--outputCol 'eprod' \
--scalingVec '2,3,4,5'


# run_feature SPFeatureHasher \
# --inputTable 'iris_tmp' \
# --outputTable 'majik_feature_featurehasher_temp' \
# --inputCols 'sepal_length,sepal_width,petal_length,petal_width,class' \
# --outputCol 'hash_features'


run_feature SPHashingTF \
--inputTable 'iris_rformula_tmp' \
--outputTable 'majik_feature_hashingtf_temp' \
--inputCol 'values' \
--outputCol 'hash_values'


run_feature SPIDF \
--inputTable 'iris_rformula_tmp' \
--outputTable 'majik_feature_idf_temp' \
--inputCol 'features' \
--outputCol 'idf_features'


run_feature SPImputer \
--inputTable 'iris_rformula_tmp' \
--outputTable 'majik_feature_imputer_temp' \
--inputCols 'sepal_length,sepal_width,petal_length,petal_width' \
--outputCols 'out_sepal_length,out_sepal_width,out_petal_length,out_petal_width'


run_feature SPIndexToString \
--inputTable 'iris_rformula_tmp' \
--outputTable 'majik_feature_indextostring_temp' \
--inputCol 'label' \
--outputCol 'string_label' \
--labels 'IndexToString1,IndexToString2,IndexToString3'


run_feature SPMaxAbsScaler \
--inputTable 'iris_rformula_tmp' \
--outputTable 'majik_feature_maxabsscaler_temp' \
--inputCol 'features' \
--outputCol 'scaled_features'


run_feature SPMinHashLSH \
--inputTable 'iris_rformula_tmp' \
--outputTable 'majik_feature_minhashlsh_temp' \
--inputCol 'features' \
--outputCol 'hashed_features'


run_feature SPMinMaxScaler \
--inputTable 'iris_rformula_tmp' \
--outputTable 'majik_feature_minmaxscaler_temp' \
--inputCol 'features' \
--outputCol 'scaled_features'


run_feature SPNGram \
--inputTable 'iris_rformula_tmp' \
--outputTable 'majik_feature_ngram_temp' \
--inputCol 'values' \
--outputCol 'ngram_values'


run_feature SPNormalizer \
--inputTable 'iris_rformula_tmp' \
--outputTable 'majik_feature_normalizer_temp' \
--inputCol 'features' \
--outputCol 'normalizer_features'


run_feature SPOneHotEncoder \
--inputTable 'iris_rformula_tmp' \
--outputTable 'majik_feature_onehotencoder_temp' \
--inputCol 'label' \
--outputCol 'onehotencoder_label'


# run_feature SPOneHotEncoderEstimator \
# --inputTable 'iris_rformula_tmp' \
# --outputTable 'majik_feature_onehotencoderestimator_temp' \
# --inputCol 'features' \
# --outputCol 'onehotencoderestimator_features'


run_feature SPPCA \
--inputTable 'iris_rformula_tmp' \
--outputTable 'majik_feature_pca_temp' \
--inputCol 'features' \
--outputCol 'pca_features' \
--k '2'


run_feature SPPolynomialExpansion \
--inputTable 'iris_rformula_tmp' \
--outputTable 'majik_feature_polynomialexpansion_temp' \
--inputCol 'features' \
--outputCol 'polynomialexpansion_features'


run_feature SPQuantileDiscretizer \
--inputTable 'iris_rformula_tmp' \
--outputTable 'majik_feature_quantilediscretizer_temp' \
--inputCol 'sepal_length' \
--outputCol 'buckets'


run_feature SPRegexTokenizer \
--inputTable 'iris_rformula_tmp' \
--outputTable 'majik_feature_regextokenizer_temp' \
--inputCol 'class' \
--outputCol 'tokens' \
--pattern 'is'


PYSPARK_PYTHON=${PY2ENV}/bin/python \
spark-submit \
--master local[*] \
--jars ${LIB_JAR} \
components/spark/ml/feature/SPSQLTransformer.py \
--inputTable 'iris_rformula_tmp' \
--outputTable 'majik_feature_sqltransformer_temp' \
--statement 'SELECT features, label FROM __THIS__'


run_feature SPStandardScaler \
--inputTable 'iris_rformula_tmp' \
--outputTable 'majik_feature_standardscaler_temp' \
--inputCol 'features' \
--outputCol 'scaled_features'


run_feature SPStopWordsRemover \
--inputTable 'iris_rformula_tmp' \
--outputTable 'majik_feature_stopwordsremover_temp' \
--inputCol 'values' \
--outputCol 'filtered_values' \
--stopWords 'a'


run_feature SPStringIndexer \
--inputTable 'iris_rformula_tmp' \
--outputTable 'majik_feature_stringindexer_temp' \
--inputCol 'class' \
--outputCol 'indexed_class'


run_feature SPTokenizer \
--inputTable 'iris_rformula_tmp' \
--outputTable 'majik_feature_tokenizer_temp' \
--inputCol 'class' \
--outputCol 'tokenizered_class'


run_feature SPVectorIndexer \
--inputTable 'iris_rformula_tmp' \
--outputTable 'majik_feature_vectorindexer_temp' \
--inputCol 'features' \
--outputCol 'indexed_features'


# run_feature SPVectorSizeHint \
# --inputTable 'iris_rformula_tmp' \
# --outputTable 'majik_feature_vectorsizehint_temp' \
# --inputCol 'features' \
# --size 4


run_feature SPVectorSlicer \
--inputTable 'iris_rformula_tmp' \
--outputTable 'majik_feature_vectorslicer_temp' \
--inputCol 'features' \
--outputCol 'sliced_features' \
--indices '1,2,3'


run_feature SPWord2Vec \
--inputTable 'iris_rformula_tmp' \
--outputTable 'majik_feature_word2vec_temp' \
--inputCol 'values' \
--outputCol 'values_model'
