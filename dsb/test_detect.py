import argparse
import os
import time
import numpy as np
from importlib import import_module
import shutil
from .utils import *
import sys
from .split_combine import SplitComb

import torch
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable

from .layers import acc

def test_detect(data_loader, net, get_pbb, save_dir, config,n_gpu):
    # Detect if the module/net use GPU.
    use_gpu = next(net.parameters()).is_cuda

    start_time = time.time()
    net.eval()
    split_comber = data_loader.dataset.split_comber
    for i_name, (data, target, coord, nzhw) in enumerate(data_loader):
        target = [np.asarray(t, np.float32) for t in target]
        for index in range(len(data)):
            s = time.time()
            lbb = target[index]
            _nzhw = nzhw[index]
            name = data_loader.dataset.filenames[index].split('-')[0].split('/')[-1]
            shortname = name.split('_clean')[0]
            shortname = os.path.split(shortname)[-1]
            _data = data[index][0]
            _coord = coord[index][0]
            isfeat = False
            if 'output_feature' in config:
                if config['output_feature']:
                    isfeat = True
            n_per_run = n_gpu
            print(_data.size())
            splitlist = list(range(0,len(_data)+1,n_gpu)) if n_gpu else lost(range(0,len(_data)+1))
            if splitlist[-1]!=len(_data):
                splitlist.append(len(_data))
            outputlist = []
            featurelist = []
            print('N-net runs on patches:', len(splitlist)-1)
            for i in range(len(splitlist)-1):
                with torch.no_grad():
                    input = Variable(_data[splitlist[i]:splitlist[i+1]])
                    inputcoord = Variable(_coord[splitlist[i]:splitlist[i+1]])
                    # Convert Variables to cuda if the module use GPU.
                    if use_gpu:
                        input = input.cuda()
                        inputcoord = inputcoord.cuda()
                    if isfeat: # TODO: no enter.
                        output,feature = net(input,inputcoord)
                        featurelist.append(feature.data.cpu().numpy())
                    else:
                        output = net(input,inputcoord) # N-Net Predict
                    outputlist.append(output.data.cpu().numpy())
            output = np.concatenate(outputlist,0)
            output = split_comber.combine(output,nzhw=_nzhw)
            if isfeat: # TODO: no enter.
                feature = np.concatenate(featurelist,0).transpose([0,2,3,4,1])[:,:,:,:,:,np.newaxis]
                feature = split_comber.combine(feature,sidelen)[...,0]

            thresh = -3
            pbb,mask = get_pbb(output,thresh,ismask=True)
            if isfeat: # TODO: no enter.
                feature_selected = feature[mask[0],mask[1],mask[2]]
                np.save(os.path.join(save_dir, shortname+'_feature.npy'), feature_selected)
            #tp,fp,fn,_ = acc(pbb,lbb,0,0.1,0.1)
            #print([len(tp),len(fp),len(fn)])
            # Print the index of patient 0~N and preprocessed npy short name path ends with the patient id.
            print([index,shortname])
            e = time.time()
            print('saving: result of {} to dir:{}'.format(shortname, save_dir))
            np.save(os.path.join(save_dir, shortname+'_pbb.npy'), pbb)
            np.save(os.path.join(save_dir, shortname+'_lbb.npy'), lbb)
    end_time = time.time()


    print('elapsed time is %3.2f seconds' % (end_time - start_time))
    print()
    print()
