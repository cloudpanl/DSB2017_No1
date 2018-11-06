# coding: utf-8
""" Preprocess DSB stage1/2 or alike data for inferencing.
(doesn't need annotation).
usage: inference_1_preprocess.py [-h] [--data DATA] [--save SAVE]
                                 [--workers WORKERS] [--reuse REUSE]

optional arguments:
  -h, --help         show this help message and exit
  --data DATA        DSB stage1/2 or similar directory path.
  --save SAVE        Directory to save preprocessed npy files to.
  --workers WORKERS  Number of workers for multi-processing
  --reuse REUSE      Reuse/skip existing preprocessed result.
"""

from dsb.preprocessing import full_prep

from multiprocessing import freeze_support

import argparse

if __name__ == '__main__':
    
    freeze_support() # Multiprocess freeze support needed only on Windows.
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='DSB stage1/2 or similar directory path.', default='F:\\LargeFiles\\DSB2017_S\\stage1_samples_sub\\')
    parser.add_argument('--save', help='Directory to save preprocessed npy files to.', default='F:\\LargeFiles\\lfz\\prep_result_sub\\')
    parser.add_argument('--workers', help='Number of workers for multi-processing', default=1)
    parser.add_argument('--reuse', help='Reuse/skip existing preprocessed result.', default=True)


    args = parser.parse_args()

    datapath = args.data
    prep_result_path = args.save 
    workers = int(args.workers)
    use_existing = args.reuse

    print('datapath:', datapath)
    print('prep_result_path:', prep_result_path)
    print('worker:', workers)

    testsplit = full_prep(datapath,prep_result_path,
                        n_worker = workers,
                        use_existing=use_existing)
    print(testsplit)
