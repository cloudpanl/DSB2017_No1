# coding: utf-8
""" Generate cancer prediction and save to a CSV file.
usage: inference_3_c_net.py [-h] [--data DATA] [--csv CSV] [--save SAVE]
                            [--bbox BBOX] [--cpkt CPKT] [--gpu GPU]

optional arguments:
  -h, --help   show this help message and exit
  --data DATA  DSB stage1/2 or similar directory path.
  --csv CSV    Save prediction CSV.
  --save SAVE  Directory to save preprocessed npy files to.
  --bbox BBOX  Path to save _pbb.npy and _lbb.npy, default be same as save dir
  --cpkt CPKT  Path to classifier.ckpt.
  --gpu GPU    Use GPU if available
"""

from dsb.data_classifier import DataBowl3Classifier
import dsb.net_classifier as casemodel
import numpy as np
import pandas
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='DSB stage1/2 or similar directory path.', default='F:\\LargeFiles\\DSB2017_S\\stage1_samples_sub\\')
    parser.add_argument('--csv', help='Save prediction CSV.', default='predicts.csv')
    parser.add_argument('--save', help='Directory to save preprocessed npy files to.', default='F:\\LargeFiles\\lfz\\prep_result_sub\\')
    parser.add_argument('--bbox', help='Path to save _pbb.npy and _lbb.npy, default be same as save dir', default=None)
    parser.add_argument('--cpkt', help='Path to classifier.ckpt.', default='./dsb/model/classifier.ckpt')
    parser.add_argument('--gpu', help='Use GPU if available', default=True)

    args = parser.parse_args()
    datapath = args.data
    prep_result_path = args.save

    filename = args.csv
    
    bbox_result_path = args.bbox
    use_gpu = args.gpu
    if not torch.cuda.is_available():
        use_gpu = False

    if bbox_result_path is None:
        bbox_result_path = prep_result_path
    
    casenet = casemodel.CaseNet(topk=5)
    checkpoint = torch.load(args.cpkt)
    casenet.load_state_dict(checkpoint['state_dict'])
    
    if use_gpu:
        casenet = casenet.cuda()
    else:
        casenet = casenet.cpu()

    testsplit = os.listdir(datapath)
    print(testsplit)

    config2 = casemodel.config

    config2['bboxpath'] = bbox_result_path
    config2['datadir'] = prep_result_path


    def test_casenet(model,testset):

        use_gpu = next(model.parameters()).is_cuda
        data_loader = DataLoader(
            testset,
            batch_size = 1,
            shuffle = False,
            num_workers = 4,
            pin_memory=True)
        model.eval()
        predlist = []

        for i,(x,coord) in enumerate(data_loader):

            coord = Variable(coord)
            x = Variable(x) 
            if use_gpu:
                coord = coord.cuda()
                x = x.cuda()
            nodulePred,casePred,_ = model(x,coord) # Predict with C-Net.
            predlist.append(casePred.data.cpu().numpy())
        predlist = np.concatenate(predlist)
        return predlist
    
    config2['bboxpath'] = bbox_result_path
    config2['datadir'] = prep_result_path

    dataset = DataBowl3Classifier(testsplit, config2, phase = 'test')
    predlist = test_casenet(casenet,dataset).T
    df = pandas.DataFrame({'id':testsplit, 'cancer':predlist})
    print('Save:,', filename)
    df.to_csv(filename,index=False)
