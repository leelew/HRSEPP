import numpy as np
import torch
import time
import os
import pandas as pd
from model.shen import CudnnLstmModel


def trainModel(model,
               x,
               y,
               lossFun,
               *,
               nEpoch=500,
               miniBatch=[100, 30],
               saveEpoch=100,
               saveFolder=None,
               mode='seq2seq',
               bufftime=0):
    batchSize, rho = miniBatch

    ngrid, nt, nx = x.shape

    if batchSize >= ngrid:
        # batchsize larger than total grids
        batchSize = ngrid

    nIterEp = int(
        np.ceil(np.log(0.01) / np.log(1 - batchSize * rho / ngrid / (nt-bufftime))))
    if hasattr(model, 'ctRm'):
        if model.ctRm is True:
            nIterEp = int(
                np.ceil(
                    np.log(0.01) / np.log(1 - batchSize *
                                          (rho - model.ct) / ngrid / (nt-bufftime))))

    if torch.cuda.is_available():
        lossFun = lossFun.cuda()
        model = model.cuda()

    optim = torch.optim.Adadelta(model.parameters())
    model.zero_grad()

    for iEpoch in range(1, nEpoch + 1):
        lossEp = 0
        t0 = time.time()
        for iIter in range(0, nIterEp):
            # training iterations
            iGrid, iT = randomIndex(ngrid, nt, [batchSize, rho])
            xTrain = selectSubset(x, iGrid, iT, rho)
            # xTrain = rho/time * Batchsize * Ninput_var
            yTrain = selectSubset(y, iGrid, iT, rho)
            # yTrain = rho/time * Batchsize * Ntraget_var
            yP = model(xTrain)
            loss = lossFun(yP, yTrain)
            loss.backward()
            optim.step()
            model.zero_grad()
            lossEp = lossEp + loss.item()
            # if iIter % 30 == 0:
            #     print('Iter {} of {}: Loss {:.3f}'.format(iIter, nIterEp, loss.item()))
        # print loss
        lossEp = lossEp / nIterEp
        logStr = 'Epoch {} Loss {:.3f} time {:.2f}'.format(
            iEpoch, lossEp,
            time.time() - t0)
        print(logStr)

    return model

class RmseLoss(torch.nn.Module):
    def __init__(self):
        super(RmseLoss, self).__init__()

    def forward(self, output, target):
        ny = target.shape[2]
        loss = 0
        for k in range(ny):
            p0 = output[:, :, k]
            t0 = target[:, :, k]
            mask = t0 == t0
            p = p0[mask]
            t = t0[mask]
            temp = torch.sqrt(((p - t)**2).mean())
            loss = loss + temp
        return loss

def train(x, y):

    lossFun = RmseLoss()
 

    model = CudnnLstmModel(
        nx=7,
        ny=1,
        hiddenSize=256)

    model = trainModel(
        model,
        x,
        y,
        lossFun,
        nEpoch=500,
        miniBatch=[100, 30])
    return model



def randomSubset(x, y, dimSubset):
    ngrid, nt, nx = x.shape
    batchSize, rho = dimSubset
    xTensor = torch.zeros([rho, batchSize, x.shape[-1]], requires_grad=False)
    yTensor = torch.zeros([rho, batchSize, y.shape[-1]], requires_grad=False)
    iGrid = np.random.randint(0, ngrid, [batchSize])
    iT = np.random.randint(0, nt - rho, [batchSize])
    for k in range(batchSize):
        temp = x[iGrid[k]:iGrid[k] + 1, np.arange(iT[k], iT[k] + rho), :]
        xTensor[:, k:k + 1, :] = torch.from_numpy(np.swapaxes(temp, 1, 0))
        temp = y[iGrid[k]:iGrid[k] + 1, np.arange(iT[k], iT[k] + rho), :]
        yTensor[:, k:k + 1, :] = torch.from_numpy(np.swapaxes(temp, 1, 0))
    if torch.cuda.is_available():
        xTensor = xTensor.cuda()
        yTensor = yTensor.cuda()
    return xTensor, yTensor


def randomIndex(ngrid, nt, dimSubset, bufftime=0):
    batchSize, rho = dimSubset
    iGrid = np.random.randint(0, ngrid, [batchSize])
    iT = np.random.randint(0+bufftime, nt - rho, [batchSize])
    return iGrid, iT


def selectSubset(x, iGrid, iT, rho, *, c=None, tupleOut=False, LCopt=False, bufftime=0):
    nx = x.shape[-1]
    nt = x.shape[1]
    if x.shape[0] == len(iGrid):   #hack
        iGrid = np.arange(0,len(iGrid))  # hack
    if (rho is not None) and (nt <= rho):
        iT.fill(0)

    batchSize = iGrid.shape[0]
    if iT is not None:
        # batchSize = iGrid.shape[0]
        xTensor = torch.zeros([rho+bufftime, batchSize, nx], requires_grad=False)
        for k in range(batchSize):
            temp = x[iGrid[k]:iGrid[k] + 1, np.arange(iT[k]-bufftime, iT[k] + rho), :]
            xTensor[:, k:k + 1, :] = torch.from_numpy(np.swapaxes(temp, 1, 0))
    else:
        if LCopt is True:
            # used for local calibration kernel: FDC, SMAP...
            if len(x.shape) == 2:
                # Used for local calibration kernel as FDC
                # x = Ngrid * Ntime
                xTensor = torch.from_numpy(x[iGrid, :]).float()
            elif len(x.shape) == 3:
                # used for LC-SMAP x=Ngrid*Ntime*Nvar
                xTensor = torch.from_numpy(np.swapaxes(x[iGrid, :, :], 1, 2)).float()
        else:
            # Used for rho equal to the whole length of time series
            xTensor = torch.from_numpy(np.swapaxes(x[iGrid, :, :], 1, 0)).float()
            rho = xTensor.shape[0]
    if c is not None:
        nc = c.shape[-1]
        temp = np.repeat(
            np.reshape(c[iGrid, :], [batchSize, 1, nc]), rho+bufftime, axis=1)
        cTensor = torch.from_numpy(np.swapaxes(temp, 1, 0)).float()

        if (tupleOut):
            if torch.cuda.is_available():
                xTensor = xTensor.cuda()
                cTensor = cTensor.cuda()
            out = (xTensor, cTensor)
        else:
            out = torch.cat((xTensor, cTensor), 2)
    else:
        out = xTensor

    if torch.cuda.is_available() and type(out) is not tuple:
        out = out.cuda()
    return out
