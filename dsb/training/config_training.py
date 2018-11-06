config = {'stage1_data_path':'F:\\LargeFiles\\lfz\\stage1_sub\\', # TODO: stage1 full
          'luna_raw':'F:\\LargeFiles\\lfz\\luna\\raw\\', # E:\\SW_WS\\github_SW\\kaggle_ndsb2017\\ where .raw files exist
          'luna_segment':'F:\\LargeFiles\\lfz\\luna\\seg-lungs-LUNA16\\',
          
          'luna_data':'F:\\LargeFiles\\lfz\\luna\\allset',
          'preprocess_result_path':'F:\\LargeFiles\\lfz\\prep_result\\',       
          
          'luna_abbr':'./detector/labels/shorter.csv',
          'luna_label':'./detector/labels/lunaqualified.csv',
          'stage1_annos_path':['./detector/labels/label_job5.csv',
                './detector/labels/label_job4_2.csv',
                './detector/labels/label_job4_1.csv',
                './detector/labels/label_job0.csv',
                './detector/labels/label_qualified.csv'],
          'bbox_path':'../detector/results/res18/bbox/',
          'preprocessing_backend':'python'
         }
