# coding: utf-8
""" Predict nodules locations, generate _pbb.npy and _lbb.npy files for each patient.
usage: inference_2_n_net.py [-h] [--data DATA] [--save SAVE] [--bbox BBOX]
                            [--cpkt CPKT] [--gpu GPU] [--gpuno GPUNO]
                            [--margin MARGIN] [--sidelen SIDELEN]

optional arguments:
  -h, --help         show this help message and exit
  --data DATA        DSB stage1/2 or similar directory path.
  --save SAVE        Directory to save preprocessed npy files to.
  --bbox BBOX        Path to save _pbb.npy and _lbb.npy, default be same as
                     save dir
  --cpkt CPKT        Path to detector.ckpt.
  --gpu GPU          Use GPU if available
  --gpuno GPUNO      Number of GPU, 1~N
  --margin MARGIN    patch margin
  --sidelen SIDELEN  patch side length
"""

from dsb.layers import acc
from dsb.data_detector import DataBowl3Detector,collate

from dsb.utils import *
from dsb.split_combine import SplitComb
from dsb.test_detect import test_detect

import torch
# from torch.nn import DataParallel
# from torch.backends import cudnn
from torch.utils.data import DataLoader


# N-net definition.
import dsb.net_detector as nodmodel
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='DSB stage1/2 or similar directory path.', default='F:\\LargeFiles\\DSB2017_S\\stage1_samples_sub\\')
    parser.add_argument('--save', help='Directory to save preprocessed npy files to.', default='F:\\LargeFiles\\lfz\\prep_result_sub\\')
    parser.add_argument('--bbox', help='Path to save _pbb.npy and _lbb.npy, default be same as save dir', default=None)
    parser.add_argument('--cpkt', help='Path to detector.ckpt.', default='./dsb/model/detector.ckpt')
    parser.add_argument('--gpu', help='Use GPU if available', default=True)
    parser.add_argument('--gpuno', help='Number of GPU, 1~N', default=1)
    parser.add_argument('--margin', help='patch margin', default=16) # TODO 32
    parser.add_argument('--sidelen', help='patch side length', default=64) # TODO 144

    args = parser.parse_args()
    datapath = args.data
    prep_result_path = args.save

    margin = args.margin
    sidelen = args.sidelen
    bbox_result_path = args.bbox
    use_gpu = args.gpu
    n_gpu = args.gpuno
    if not torch.cuda.is_available():
        use_gpu = False

    if bbox_result_path is None:
        bbox_result_path = prep_result_path
    
    os.makedirs(bbox_result_path, exist_ok=True)

    # This loads net_detector.py file as a module.
    # nodmodel = import_module('dsb.net_detector')
    config1, nod_net, loss, get_pbb = nodmodel.get_model()
    checkpoint = torch.load(args.cpkt)
    nod_net.load_state_dict(checkpoint['state_dict'])

    if use_gpu:
        nod_net= nod_net.cuda()
    else:
        nod_net= nod_net.cpu()
    testsplit = os.listdir(datapath)
    print(testsplit)


    config1['datadir'] = prep_result_path
    split_comber = SplitComb(sidelen,config1['max_stride'],config1['stride'],margin,pad_value= config1['pad_value'])

    dataset = DataBowl3Detector(testsplit,config1,phase='test',split_comber=split_comber)
    test_loader = DataLoader(dataset,batch_size = 1,
        shuffle = False,num_workers = 4,pin_memory=False,collate_fn =collate)

    test_detect(test_loader, nod_net, get_pbb, bbox_result_path,config1,n_gpu=n_gpu)
