"""N-Net training
usage: train_2_n_net.py [-h] [--preprocess_result_path PREPROCESS_RESULT_PATH]
                        [-j N] [--epochs N] [--start_epoch N] [-b N] [--lr LR]
                        [--momentum M] [--weight_decay W] [--save_freq S]
                        [--resume PATH] [--save_dir SAVE] [--test TEST]
                        [--split SPLIT] [--gpu N] [--n_test N]
                        [--train_ids TRAIN_IDS] [--val_ids VAL_IDS]
                        [--test_ids TEST_IDS]

N-Net training

optional arguments:
  -h, --help            show this help message and exit
  --preprocess_result_path PREPROCESS_RESULT_PATH
                        Directory to save preprocessed _clean and _label .npy
                        files.
  -j N, --workers N     number of data loading workers (default: 32)
  --epochs N            number of total epochs to run
  --start_epoch N       manual epoch number (useful on restarts)
  -b N, --batch-size N  mini-batch size (default: 16)
  --lr LR, --learning-rate LR
                        initial learning rate
  --momentum M          momentum
  --weight_decay W, --wd W
                        weight decay (default: 1e-4)
  --save_freq S         save frequency
  --resume PATH         path to latest checkpoint (default: none)
  --save_dir SAVE       directory to save checkpoint (default: none)
  --test TEST           1 do test evaluation, 0 not
  --split SPLIT         In the test phase, split the image to 8 parts
  --gpu N               use gpu, set to `none` to use CPU
  --n_test N            number of gpu for test
  --train_ids TRAIN_IDS
                        Path to the npy file for training scan IDs stored in a
                        Numpy list.
  --val_ids VAL_IDS     Path to the npy file for validation scan IDs stored in
                        a Numpy list.
  --test_ids TEST_IDS   Path to the npy file for test scan IDs stored in a
                        Numpy list.

Usage Example:
python train_2_n_net.py -b 1 --epochs 001 --save_dir res18 -j 1 --save_freq 1
python train_2_n_net.py -b 1 --epochs 002 --save_dir res18 -j 1 --save_freq 1 --resume results/res18/001.ckpt --gpu all
"""

from multiprocessing import freeze_support
import argparse
import os
import time
import numpy as np
import dsb.training.detector.data as data
import shutil
from dsb.training.detector.utils import *
import sys
from dsb.training.detector.split_combine import SplitComb

import torch
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable

from dsb.training.detector.layers import acc
import dsb.training.detector.res18 as model

def main():
    parser = argparse.ArgumentParser(description='N-Net training')
    parser.add_argument('--preprocess_result_path', help='Directory to save preprocessed _clean and _label .npy files.',
                        default='F:\\LargeFiles\\lfz\\prep_result_sub\\')
    parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=16, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--save_freq', default='10', type=int, metavar='S',
                        help='save frequency')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--save_dir', default='', type=str, metavar='SAVE',
                        help='directory to save checkpoint (default: none)')
    parser.add_argument('--test', default=0, type=int, metavar='TEST',
                        help='1 do test evaluation, 0 not')
    parser.add_argument('--split', default=8, type=int, metavar='SPLIT',
                        help='In the test phase, split the image to 8 parts')
    parser.add_argument('--gpu', default='all', type=str, metavar='N',
                        help='use gpu, set to `none` to use CPU')
    parser.add_argument('--n_test', default=8, type=int, metavar='N',
                        help='number of gpu for test')
    parser.add_argument('--train_ids', default='./dsb/training/detector/kaggleluna_full.npy', type=str,
                        help='Path to the npy file for training scan IDs stored in a Numpy list.')
    parser.add_argument('--val_ids', default='./dsb/training/detector/kaggleluna_full.npy', type=str, # TODO: replace with valsplit.npy when full datasets available.
                        help='Path to the npy file for validation scan IDs stored in a Numpy list.')
    parser.add_argument('--test_ids', default='./dsb/training/detector/full.npy', type=str,
                        help='Path to the npy file for test scan IDs stored in a Numpy list.')
    args = parser.parse_args()

    torch.manual_seed(0)
    use_gpu = False
    if 'none' not in args.gpu.lower() and torch.cuda.is_available():
        use_gpu = True
        torch.cuda.set_device(0)

    config, net, loss, get_pbb = model.get_model()
    start_epoch = args.start_epoch
    save_dir = args.save_dir

    if args.resume:
        checkpoint = torch.load(args.resume)
        if start_epoch == 0:
            start_epoch = checkpoint['epoch'] + 1
        if not save_dir:
            save_dir = checkpoint['save_dir']
        else:
            save_dir = os.path.join('results',save_dir)
        net.load_state_dict(checkpoint['state_dict'])
    else:
        if start_epoch == 0:
            start_epoch = 1
        if not save_dir:
            exp_id = time.strftime('%Y%m%d-%H%M%S', time.localtime())
            save_dir = os.path.join('results', args.model + '-' + exp_id)
        else:
            save_dir = os.path.join('results',save_dir)

    os.makedirs(save_dir, exist_ok=True)

    logfile = os.path.join(save_dir,'log')
    if use_gpu:
        print('Use GPU for training.')
        n_gpu = setgpu(args.gpu)
        args.n_gpu = n_gpu
        net = net.cuda()
        loss = loss.cuda()
        cudnn.benchmark = True
        net = DataParallel(net)
    else:
        print('Use CPU for training.')
        net = net.cpu()
    datadir = args.preprocess_result_path

    if args.test == 1:
        margin = 32
        sidelen = 144

        split_comber = SplitComb(sidelen,config['max_stride'],config['stride'],margin,config['pad_value'])
        # Test sets.
        dataset = data.DataBowl3Detector(
            datadir,
            args.test_ids,
            config,
            phase='test',
            split_comber=split_comber)
        test_loader = DataLoader(
            dataset,
            batch_size = 1,
            shuffle = False,
            num_workers = args.workers,
            collate_fn = data.collate,
            pin_memory=False)

        test(test_loader, net, get_pbb, save_dir,config, args)
        return

    # Train sets
    dataset = data.DataBowl3Detector(
        datadir,
        args.train_ids,
        config,
        phase = 'train')

    print('batch_size:', args.batch_size)
    train_loader = DataLoader(
        dataset,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = args.workers,
        pin_memory=True)

    # Validation sets
    dataset = data.DataBowl3Detector(
        datadir,
        args.val_ids,
        config,
        phase = 'val')
    val_loader = DataLoader(
        dataset,
        batch_size = args.batch_size,
        shuffle = False,
        num_workers = args.workers,
        pin_memory=True)

    optimizer = torch.optim.SGD(
        net.parameters(),
        args.lr,
        momentum = 0.9,
        weight_decay = args.weight_decay)

    def get_lr(epoch):
        if epoch <= args.epochs * 0.5:
            lr = args.lr
        elif epoch <= args.epochs * 0.8:
            lr = 0.1 * args.lr
        else:
            lr = 0.01 * args.lr
        return lr


    for epoch in range(start_epoch, args.epochs + 1):
        train(train_loader, net, loss, epoch, optimizer, get_lr, args.save_freq, save_dir, args)
        validate(val_loader, net, loss)

def train(data_loader, net, loss, epoch, optimizer, get_lr, save_freq, save_dir):
    # Check if the net use GPU.
    use_gpu = next(net.parameters()).is_cuda
    start_time = time.time()

    net.train()
    lr = get_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    metrics = []
    for i, (data, target, coord) in enumerate(data_loader):
        data = Variable(data)
        target = Variable(target)
        coord = Variable(coord)
        if use_gpu:
            data = data.cuda(async = True)
            target = target.cuda(async = True)
            coord = coord.cuda(async = True)

        output = net(data, coord)
        loss_output = loss(output, target)

        optimizer.zero_grad()
        loss_output[0].backward()
        optimizer.step()

        loss_output[0] = loss_output[0].item()
        metrics.append(loss_output)

    if epoch % save_freq == 0:
        if isinstance(net, DataParallel):
            state_dict = net.module.state_dict()
        else:
            state_dict = net.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()

        torch.save({
            'epoch': epoch,
            'save_dir': save_dir,
            'state_dict': state_dict},
            os.path.join(save_dir, '%03d.ckpt' % epoch))

    end_time = time.time()

    metrics = np.asarray(metrics, np.float32)
    print('Epoch %03d (lr %.5f)' % (epoch, lr))
    print('Train:      tpr %3.2f, tnr %3.2f, total pos %d, total neg %d, time %3.2f' % (
        100.0 * np.sum(metrics[:, 6]) / np.sum(metrics[:, 7]),
        100.0 * np.sum(metrics[:, 8]) / np.sum(metrics[:, 9]),
        np.sum(metrics[:, 7]),
        np.sum(metrics[:, 9]),
        end_time - start_time))
    print('loss %2.4f, classify loss %2.4f, regress loss %2.4f, %2.4f, %2.4f, %2.4f' % (
        np.mean(metrics[:, 0]),
        np.mean(metrics[:, 1]),
        np.mean(metrics[:, 2]),
        np.mean(metrics[:, 3]),
        np.mean(metrics[:, 4]),
        np.mean(metrics[:, 5])))
    print

def validate(data_loader, net, loss):
    # Check if the net use GPU.
    use_gpu = next(net.parameters()).is_cuda
    start_time = time.time()

    net.eval()

    metrics = []
    for i, (data, target, coord) in enumerate(data_loader):
        with torch.no_grad():
            data = Variable(data)
            target = Variable(target)
            coord = Variable(coord)
            if use_gpu:
                data = data.cuda(async = True)
                target = target.cuda(async = True)
                coord = coord.cuda(async = True)

            output = net(data, coord)
            loss_output = loss(output, target, train = False)

            loss_output[0] = loss_output[0].item()
            metrics.append(loss_output)
    end_time = time.time()

    metrics = np.asarray(metrics, np.float32)
    print('Validation: tpr %3.2f, tnr %3.8f, total pos %d, total neg %d, time %3.2f' % (
        100.0 * np.sum(metrics[:, 6]) / np.sum(metrics[:, 7]),
        100.0 * np.sum(metrics[:, 8]) / np.sum(metrics[:, 9]),
        np.sum(metrics[:, 7]),
        np.sum(metrics[:, 9]),
        end_time - start_time))
    print('loss %2.4f, classify loss %2.4f, regress loss %2.4f, %2.4f, %2.4f, %2.4f' % (
        np.mean(metrics[:, 0]),
        np.mean(metrics[:, 1]),
        np.mean(metrics[:, 2]),
        np.mean(metrics[:, 3]),
        np.mean(metrics[:, 4]),
        np.mean(metrics[:, 5])))
    print()
    print()

def test(data_loader, net, get_pbb, save_dir, config, args):
    # Check if the net use GPU.
    use_gpu = next(net.parameters()).is_cuda
    start_time = time.time()
    save_dir = os.path.join(save_dir,'bbox')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(save_dir)
    net.eval()
    namelist = []
    split_comber = data_loader.dataset.split_comber
    for i_name, (data, target, coord, nzhw) in enumerate(data_loader):
        s = time.time()
        target = [np.asarray(t, np.float32) for t in target]
        lbb = target[0]
        nzhw = nzhw[0]
        name = data_loader.dataset.filenames[i_name].split('-')[0].split('/')[-1].split('_clean')[0]
        data = data[0][0]
        coord = coord[0][0]
        isfeat = False
        if 'output_feature' in config:
            if config['output_feature']:
                isfeat = True
        n_per_run = args.n_test
        print(data.size())
        splitlist = list(range(0,len(data)+1,n_per_run))
        if splitlist[-1]!=len(data):
            splitlist.append(len(data))
        outputlist = []
        featurelist = []

        for i in range(len(splitlist)-1):
            with torch.no_grad():
                input = Variable(data[splitlist[i]:splitlist[i+1]])
                inputcoord = Variable(coord[splitlist[i]:splitlist[i+1]])
                if use_gpu:
                    input = input.cuda()
                    inputcoord = inputcoord.cuda()
                if isfeat:
                    output,feature = net(input,inputcoord)
                    featurelist.append(feature.data.cpu().numpy())
                else:
                    output = net(input,inputcoord)
                outputlist.append(output.data.cpu().numpy())
        output = np.concatenate(outputlist,0)
        output = split_comber.combine(output,nzhw=nzhw)
        if isfeat:
            feature = np.concatenate(featurelist,0).transpose([0,2,3,4,1])[:,:,:,:,:,np.newaxis]
            feature = split_comber.combine(feature,sidelen)[...,0]

        thresh = -3
        pbb,mask = get_pbb(output,thresh,ismask=True)
        if isfeat:
            feature_selected = feature[mask[0],mask[1],mask[2]]
            np.save(os.path.join(save_dir, name+'_feature.npy'), feature_selected)
        #tp,fp,fn,_ = acc(pbb,lbb,0,0.1,0.1)
        #print([len(tp),len(fp),len(fn)])
        print([i_name,name])
        e = time.time()
        np.save(os.path.join(save_dir, name+'_pbb.npy'), pbb)
        np.save(os.path.join(save_dir, name+'_lbb.npy'), lbb)
    np.save(os.path.join(save_dir, 'namelist.npy'), namelist)
    end_time = time.time()


    print('elapsed time is %3.2f seconds' % (end_time - start_time))
    print
    print

if __name__ == '__main__':
    main()
