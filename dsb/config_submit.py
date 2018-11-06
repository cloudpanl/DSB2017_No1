config = {'datapath':'F:\\LargeFiles\\DSB2017_S\\stage1_samples_sub\\', # TODO: stage2
          'preprocess_result_path':'F:\\LargeFiles\\lfz\\prep_result_sub\\',
          'outputfile':'prediction.csv',
          
          'detector_model':'net_detector',
         'detector_param':'./model/detector.ckpt',
         'classifier_model':'net_classifier',
         'classifier_param':'./model/classifier.ckpt',
         'n_gpu':1, # 1~N, cannot be set to 0 since it is used to calculate the input size.
         'n_worker_preprocessing':1, # TODO: 4
         'use_exsiting_preprocessing':True,
         'skip_preprocessing':True,
         'skip_detect':True}
