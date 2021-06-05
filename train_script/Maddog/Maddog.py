import sys, os
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd
import numpy as np
import pickle
from IPython import embed

# Used to initilize critic weights, since always cliped between -0.01 and 0.01
def weights_init_clamped(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.uniform_(m.weight.data, -0.01, 0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
            
# The network used for the generator
class NetG(nn.Module):
    def __init__(self, featLen, repLen):
        super(NetG, self).__init__()
        self.CONV1 = nn.Conv1d(in_channels=featLen, out_channels=repLen, kernel_size=15, bias=False, padding=7)
        self.CONV2 = nn.Conv1d(in_channels=repLen, out_channels=repLen, kernel_size=5, bias=False, dilation=2, padding=2+2)
        self.POOL = nn.AdaptiveMaxPool1d(1)
        self.DROP = nn.Dropout(p=0.2)

    def forward(self, h):
        h = self.CONV1(h)
        h = F.relu(h)
        h = self.CONV2(h)
        h = F.relu(h)
        h = self.POOL(h)
        h = torch.squeeze(h, 2)
        h = self.DROP(h)
        return h
        
# The network used for the classifier and critic
class NetD(nn.Module):
    def __init__(self, repLen, numOut):
        super(NetD, self).__init__()
        self.numOut = numOut
        self.numFC = 3
        
        self.FC = nn.ModuleList()
        for lOn in range(self.numFC):
            if lOn+1 == self.numFC:
                self.FC.append(nn.Linear(in_features=repLen, out_features=numOut, bias=False))
            else:
                self.FC.append(nn.Linear(in_features=repLen, out_features=repLen, bias=False))
             
    def forward(self, h, useLast=True):
        lastNum = self.numFC
        if not useLast:
            lastNum -= 1   
        for lOn in range(lastNum):
            h = self.FC[lOn](h)
            if lOn+1 != self.numFC:
                h = F.relu(h)
        return h

# Converts data to run in PyTorch        
def ToVariable(data, requires_grad=False, cuda=True):
    data = torch.FloatTensor(data)
    data = Variable(data, requires_grad=requires_grad)
    if cuda:
        data = data.cuda()
    return data

# A weighted version of cross entropy used in model loss
def softCrossEntropy(pred, soft_targets, weights):
    logsoftmax = nn.LogSoftmax(dim=-1)
    return -torch.mean(torch.sum(soft_targets * logsoftmax(pred) * weights, -1))

class Maddog:        
    def __init__(self, featLen=None, wDs=None, wEmo=None, args=None, ptPath=None):
        # Parse arguments
        if ptPath is None:
            self.featLen = featLen
            self.wDs = wDs
            self.wEmo = ToVariable(torch.from_numpy(wEmo).float())
            self.args = args
        else:
            with open(os.path.join(ptPath, 'args.p'), 'rb') as f:
                vals = pickle.load(f)
            self.featLen = vals['featLen']
            self.wDs = vals['wDs']
            self.wEmo = vals['wEmo']
            self.args = vals['args']
            
        # Setup nets
        self.netG = NetG(self.featLen, self.args.repLen)
        self.netD = NetD(self.args.repLen, len(self.wDs))
        for m in self.netD.modules():
            weights_init_clamped(m)
        self.netC = NetD(self.args.repLen, len(self.wEmo))
          
        # Send to CUDA and get params
        self.netG.cuda()
        pListG = []
        pListG.extend(list(self.netG.parameters()))
        self.netD.cuda()
        self.netC.cuda()
        pListG.extend(list(self.netC.parameters()))

        # Setup optimizers
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=self.args.learningRate, eps=1e-08)
        self.optimizerG = optim.Adam(pListG, lr=self.args.learningRate, eps=1e-08)

    def Save(self, outPath):
        try:
            os.makedirs(outPath)
        except OSError as exc:
            pass
        vals = {}
        vals['featLen'] = self.featLen
        vals['wDs'] = self.wDs
        vals['wEmo'] = self.wEmo
        vals['args'] = self.args
        with open(os.path.join(outPath, 'args.p'), 'wb') as f:
            pickle.dump(vals, f)
        torch.save(self.netG.state_dict(), os.path.join(outPath, 'G.pt'))
        torch.save(self.netD.state_dict(), os.path.join(outPath, 'D.pt'))
        torch.save(self.netC.state_dict(), os.path.join(outPath, 'C.pt'))
        
    def Load(self, outPath):
        self.netG.load_state_dict(torch.load(os.path.join(outPath, 'G.pt')))
        self.netD.load_state_dict(torch.load(os.path.join(outPath, 'D.pt')))
        self.netC.load_state_dict(torch.load(os.path.join(outPath, 'C.pt')))
        
    def Fit(self, itTrn, itTrnEmo):
        numBatches = int(len(itTrn) / self.args.batchSize)
        self.netG.train()
        self.netC.train()

        for iteration in range(numBatches):
            ############################
            # (1) Update D network
            ###########################
            for p in self.netD.parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in netG update

            for iter_d in range(self.args.dRepeats):
                self.netD.zero_grad()
            
                # Train Critic
                batch = next(itTrn)
                features = ToVariable(torch.from_numpy(batch['Features']).float())
                isDs = batch['Dataset']
                isDs = isDs * -self.wDs
                isDs[isDs==0] = 1.0
                isDs = ToVariable(torch.from_numpy(isDs).float())
                h = autograd.Variable(self.netG(features).data)
                D = self.netD(h)
                loss = (D*isDs).mean()
                loss.backward()
                self.optimizerD.step()

                # Clamp D
                for p in self.netD.parameters():  # reset requires_grad
                    p.data.clamp_(-0.01, 0.01)
                    
            ############################
            # (2) Update G network
            ###########################
            for p in self.netD.parameters():
                p.requires_grad = False  # to avoid computation
            self.optimizerG.zero_grad()
            loss = 0.0
            
            # TRAIN WITH UNLABELED
            batch = next(itTrn)
            features = ToVariable(torch.from_numpy(batch['Features']).float())
            isDs = ToVariable(torch.from_numpy(batch['Dataset']).float())
            
            h = self.netG(features)
            D = self.netD(h)
            loss += (D*isDs).mean() * self.args.gFactor
            
            # TRAIN WITH LABELED
            batch = next(itTrnEmo)
            features = ToVariable(torch.from_numpy(batch['Features']).float())
            labels = ToVariable(torch.from_numpy(batch['Emotion']).float())
            isDs = ToVariable(torch.from_numpy(batch['Dataset']).float())
                
            h = self.netG(features)
            D = self.netD(h)
            loss += (D*isDs).mean() * self.args.gFactor
            curOut = self.netC(h)
            loss += softCrossEntropy(curOut, labels, self.wEmo)            

            # Back propagate
            loss.backward()
            self.optimizerG.step()
                        
    def Predict(self, dlVal, useLast=True):
        self.netG.eval()
        self.netC.eval()
            
        out = []
        for batch in dlVal:
            features = ToVariable(torch.from_numpy(batch['Features']).float())
            tmpOut = self.netC(self.netG(features), useLast=useLast)
            tmpOut = tmpOut.data.cpu().numpy()
            out.append(tmpOut)    
        out = np.concatenate(out, axis=0)
        return out
