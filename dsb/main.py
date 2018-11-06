from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from preprocessing import full_prep
from config_submit import config as config_submit

import torch
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable

from layers import acc
from data_detector import DataBowl3Detector,collate
from data_classifier import DataBowl3Classifier

from utils import *
from split_combine import SplitComb
from test_detect import test_detect
from importlib import import_module
import pandas
from multiprocessing import freeze_support


if __name__ == '__main__':
    freeze_support() # Multiprocess freeze support needed only on Windows.
    # Load configs from file.
    datapath = config_submit['datapath']
    prep_result_path = config_submit['preprocess_result_path']
    skip_prep = config_submit['skip_preprocessing']
    skip_detect = config_submit['skip_detect']

    # 1. Preprocess
    # testsplit is a list of patient IDs
    # The pre-processed results will be pairs of _clean, and _label npy files for a patient id.
    if not skip_prep:
        testsplit = full_prep(datapath,prep_result_path,
                            n_worker = config_submit['n_worker_preprocessing'],
                            use_existing=config_submit['use_exsiting_preprocessing'])
    else:
        testsplit = os.listdir(datapath)

    # 2. Load and run the N-net
    nodmodel = import_module(config_submit['detector_model'].split('.py')[0])
    config1, nod_net, loss, get_pbb = nodmodel.get_model()
    checkpoint = torch.load(config_submit['detector_param'])
    nod_net.load_state_dict(checkpoint['state_dict'])

    # TODO: run on CPU for now.
    # torch.cuda.set_device(0)
    # nod_net = nod_net.cuda()
    # TODO: remove this to run on GPU.
    nod_net = nod_net.cpu()
    cudnn.benchmark = True
    # TODO: temporarily remove dataparallel
    # nod_net = DataParallel(nod_net)

    bbox_result_path = './bbox_result'
    if not os.path.exists(bbox_result_path):
        os.mkdir(bbox_result_path)
    #testsplit = [f.split('_clean')[0] for f in os.listdir(prep_result_path) if '_clean' in f]

    if not skip_detect:
        margin = 16 # TODO 32
        sidelen = 64 # TODO 144
        config1['datadir'] = prep_result_path
        split_comber = SplitComb(sidelen,config1['max_stride'],config1['stride'],margin,pad_value= config1['pad_value'])

        dataset = DataBowl3Detector(testsplit,config1,phase='test',split_comber=split_comber)
        test_loader = DataLoader(dataset,batch_size = 1,
            shuffle = False,num_workers = 4,pin_memory=False,collate_fn =collate)

        test_detect(test_loader, nod_net, get_pbb, bbox_result_path,config1,n_gpu=config_submit['n_gpu'])

        

    # 3. Load and run the C-net
    casemodel = import_module(config_submit['classifier_model'].split('.py')[0])
    casenet = casemodel.CaseNet(topk=5)
    config2 = casemodel.config
    checkpoint = torch.load(config_submit['classifier_param'])
    # TODO: checkpoint converted from py2, key in binary encoding
    casenet.load_state_dict(checkpoint['state_dict'])
    
    # TODO: run on CPU for now.
    # torch.cuda.set_device(0)
    # casenet = casenet.cuda()
    # TODO: remove this to run on GPU.
    casenet = casenet.cpu()
    cudnn.benchmark = True
    
    # TODO: temporarily remove dataparallel
    # casenet = DataParallel(casenet)

    filename = config_submit['outputfile']

    def test_casenet(model,testset):
        data_loader = DataLoader(
            testset,
            batch_size = 1,
            shuffle = False,
            num_workers = 4,
            pin_memory=True)
        #model = model.cuda()
        model.eval()
        predlist = []
        
        #     weight = torch.from_numpy(np.ones_like(y).float().cuda()
        for i,(x,coord) in enumerate(data_loader):

            coord = Variable(coord) # TODO .cuda()
            x = Variable(x) # TODO .cuda()
            nodulePred,casePred,_ = model(x,coord)
            predlist.append(casePred.data.cpu().numpy())
            #print([i,data_loader.dataset.split[i,1],casePred.data.cpu().numpy()])
        predlist = np.concatenate(predlist)
        return predlist    
    config2['bboxpath'] = bbox_result_path
    config2['datadir'] = prep_result_path



    dataset = DataBowl3Classifier(testsplit, config2, phase = 'test')
    predlist = test_casenet(casenet,dataset).T
    df = pandas.DataFrame({'id':testsplit, 'cancer':predlist})
    df.to_csv(filename,index=False)
