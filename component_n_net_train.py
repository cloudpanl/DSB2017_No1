# coding: utf-8
from __future__ import print_function

import functools
import os
import time

import numpy as np
import torch
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
import torch.utils.data.distributed

import dsb.training.detector.data as data
import dsb.training.detector.res18 as model
from dsb.training.detector.utils import *
from suanpan import asyncio
from suanpan.arguments import Float, Int, String, Bool
from suanpan.docker import DockerComponent as dc
from suanpan.docker.arguments import Checkpoint, Folder, HiveTable


def getLearningRate(epoch, epochs, lr):
    if epoch <= epochs * 0.5:
        return lr
    elif epoch <= epochs * 0.8:
        return 0.1 * lr
    else:
        return 0.01 * lr


def train(data_loader, net, loss, epoch, optimizer, get_lr, save_dir):
    # Check if the net use GPU.
    use_gpu = next(net.parameters()).is_cuda
    start_time = time.time()

    net.train()
    lr = get_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    metrics = []
    for i, (data, target, coord) in enumerate(data_loader):
        data = Variable(data)
        target = Variable(target)
        coord = Variable(coord)
        if use_gpu:
            data = data.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            coord = coord.cuda(non_blocking=True)

        output = net(data, coord)
        loss_output = loss(output, target)

        optimizer.zero_grad()
        loss_output[0].backward()
        optimizer.step()

        # loss_output[0] = loss_output[0].item()
        metrics.append(loss_output)

    end_time = time.time()

    metrics = np.asarray(metrics, np.float32)
    print("Epoch %03d (lr %.5f)" % (epoch, lr))
    print(
        "Train:      tpr %3.2f, tnr %3.2f, total pos %d, total neg %d, time %3.2f"
        % (
            100.0 * np.sum(metrics[:, 6]) / np.sum(metrics[:, 7]),
            100.0 * np.sum(metrics[:, 8]) / np.sum(metrics[:, 9]),
            np.sum(metrics[:, 7]),
            np.sum(metrics[:, 9]),
            end_time - start_time,
        )
    )
    print(
        "loss %2.4f, classify loss %2.4f, regress loss %2.4f, %2.4f, %2.4f, %2.4f"
        % (
            np.mean(metrics[:, 0]),
            np.mean(metrics[:, 1]),
            np.mean(metrics[:, 2]),
            np.mean(metrics[:, 3]),
            np.mean(metrics[:, 4]),
            np.mean(metrics[:, 5]),
        )
    )


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
                data = data.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
                coord = coord.cuda(non_blocking=True)

            output = net(data, coord)
            loss_output = loss(output, target, train=False)

            # loss_output[0] = loss_output[0].item()
            metrics.append(loss_output)
    end_time = time.time()

    metrics = np.asarray(metrics, np.float32)
    print(
        "Validation: tpr %3.2f, tnr %3.8f, total pos %d, total neg %d, time %3.2f"
        % (
            100.0 * np.sum(metrics[:, 6]) / np.sum(metrics[:, 7]),
            100.0 * np.sum(metrics[:, 8]) / np.sum(metrics[:, 9]),
            np.sum(metrics[:, 7]),
            np.sum(metrics[:, 9]),
            end_time - start_time,
        )
    )
    print(
        "loss %2.4f, classify loss %2.4f, regress loss %2.4f, %2.4f, %2.4f, %2.4f"
        % (
            np.mean(metrics[:, 0]),
            np.mean(metrics[:, 1]),
            np.mean(metrics[:, 2]),
            np.mean(metrics[:, 3]),
            np.mean(metrics[:, 4]),
            np.mean(metrics[:, 5]),
        )
    )


def save(path, net, **kwargs):
    if isinstance(net, DataParallel):
        state_dict = net.module.state_dict()
    else:
        state_dict = net.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()

    kwargs.update(state_dict=state_dict)
    torch.save(kwargs, path)
    return path


@dc.input(
    HiveTable(
        key="inputTrainData",
        table="inputTrainDataTable",
        partition="inputTrainDataPartition",
    )
)
@dc.input(
    HiveTable(
        key="inputValidateData",
        table="inputValidateDataTable",
        partition="inputValidateDataPartition",
    )
)
@dc.input(Folder(key="inputDataFolder", required=True))
@dc.input(Checkpoint(key="inputCheckpoint"))
@dc.output(Checkpoint(key="outputCheckpoint", required=True))
@dc.column(String(key="idColumn", default="id"))
@dc.param(Int(key="epochs", default=100))
@dc.param(Int(key="batchSize", default=16))
@dc.param(Float(key="learningRate", default=0.01))
@dc.param(Float(key="momentum", default=0.9))
@dc.param(Float(key="weightDecay", default=1e-4))
@dc.param(Int(key="worldSize", default=1))
@dc.param(String(key="distBackend", default="nccl"))
@dc.param(String(key="distUrl", default="env://"))
@dc.param(Int(key="distRank", default=1))
def SPNNetTrain(context):
    torch.manual_seed(0)

    args = context.args

    saveFolder = os.path.dirname(args.outputCheckpoint)
    dataFolder = args.inputDataFolder
    checkoutPointPath = args.inputCheckpoint
    trainIds = args.inputTrainData[args.idColumn]
    validateIds = args.inputValidateData[args.idColumn]
    workers = asyncio.WORKERS
    epochs = args.epochs
    batchSize = args.batchSize
    learningRate = args.learningRate
    momentum = args.momentum
    weightDecay = args.weightDecay
    useGpu = torch.cuda.is_available()
    distributed = args.worldSize > 1

    if distributed:
        torch.distributed.init_process_group(
            backend=args.distBackend,
            init_method=args.distUrl,
            world_size=args.worldSize,
            rank=args.distRank,
        )

    config, net, loss, getPbb = model.get_model()

    if checkoutPointPath:
        checkpoint = torch.load(checkoutPointPath)
        startEpoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["state_dict"])
    else:
        startEpoch = 1

    if useGpu:
        print("Use GPU {} for training.".format(torch.cuda.current_device()))
        net = net.cuda()
        loss = loss.cuda()
        cudnn.benchmark = True
        net = DistributedDataParallel(net) if distributed else DataParallel(net)
    else:
        print("Use CPU for training.")
        net = net.cpu()

    # Train sets
    trainDataset = data.DataBowl3Detector(dataFolder, trainIds, config, phase="train")
    trainSampler = (
        torch.utils.data.distributed.DistributedSampler(trainDataset)
        if distributed
        else None
    )
    trainLoader = DataLoader(
        trainDataset,
        batch_size=batchSize,
        shuffle=(trainSampler is None),
        num_workers=workers,
        pin_memory=True,
        sampler=trainSampler,
    )

    # Validation sets
    valDataset = data.DataBowl3Detector(dataFolder, validateIds, config, phase="val")
    valLoader = DataLoader(
        valDataset,
        batch_size=batchSize,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    optimizer = torch.optim.SGD(
        net.parameters(), learningRate, momentum=momentum, weight_decay=weightDecay
    )

    getlr = functools.partial(getLearningRate, epochs=epochs, lr=learningRate)

    for epoch in range(startEpoch, epochs + 1):
        if trainSampler is not None:
            trainSampler.set_epoch(epoch)
        train(trainLoader, net, loss, epoch, optimizer, getlr, saveFolder)
        validate(valLoader, net, loss)

    ckptPath = os.path.join(saveFolder, "model.ckpt")
    save(ckptPath, net, epochs=epochs)

    return ckptPath


if __name__ == "__main__":
    SPNNetTrain()  # pylint: disable=no-value-for-parameter
