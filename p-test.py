import pickle
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import os
import time
import sklearn.metrics as metrics
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm as tqdm

os.chdir("/z/lancelcy/PRIORI_Emotion/mfbs")

def zero_pad_to_length(data, length=3000):
    padAm = length - data.shape[-1]
    if padAm == 0:
        return data
    else:
        return np.pad(data, ((0,0),(0,padAm)), 'constant')


def load_pk(file_stub):
    filename = file_stub
    if not os.path.exists(filename):
        return None
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
        return obj

class Net(nn.Module):   
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv1d(in_channels=40, out_channels=128, kernel_size=15, bias=False, padding=7),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, bias=False, dilation=2, padding=2+2),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(1),
            nn.Dropout(p=0.2),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(128,128),
            nn.Linear(128, 3)
        )

    # Defining the forward pass    
    def forward(self, x):
        h = self.cnn_layers(x)
        h = h.view(h.size(0), -1)
#         print(x.shape)
        h = self.linear_layers(h)
        return h

def main():
    print("loading data")
    with open("/home/lancelcy/PRIORI/X_R21.pk","rb") as f:
    	X_R21=pickle.load(f)
    with open("/home/lancelcy/PRIORI/Y_R21.pk","rb") as f:
    	Y_R21=pickle.load(f)
    print(X_R21.shape)
    print(Y_R21.shape)
    with open("/home/lancelcy/PRIORI/X_train.pk","rb") as f:
    	X_train=pickle.load(f)
    with open("/home/lancelcy/PRIORI/Y_train.pk","rb") as f:
    	Y_train=pickle.load(f)
    with open("/home/lancelcy/PRIORI/X_val.pk","rb") as f:
    	X_val=pickle.load(f)
    with open("/home/lancelcy/PRIORI/Y_val.pk","rb") as f:
    	Y_val=pickle.load(f)
    with open("/home/lancelcy/PRIORI/X_test.pk","rb") as f:
    	X_test=pickle.load(f)
    with open("/home/lancelcy/PRIORI/Y_test.pk","rb") as f:
    	Y_test=pickle.load(f)
    print("data loaded")
    train_data=TensorDataset(torch.FloatTensor(X_train),torch.LongTensor(Y_train))
    val_data=TensorDataset(torch.FloatTensor(X_val),torch.LongTensor(Y_val))
    # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
	# valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=True, num_workers=4)
    train_loader=DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader=DataLoader(val_data, batch_size=64, shuffle=True)


    # model.cuda()
    # defining the optimizer
    # defining the loss function





    # checking if GPU is available


    n_epochs = 17
    # empty list to store training losses
    test1_accuracy=[]
    test1_UAR=[]
    test2_accuracy=[]
    test2_UAR=[]
    # training the model
    for i in tqdm(range(100)):

        #initializing
        model = Net()
        train_loss=[]
        valid_loss=[]
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001,weight_decay=1e-6)
        loss_min = np.inf
        if torch.cuda.is_available():
            model = model.cuda(0)
            criterion = criterion.cuda(0)


        #training
        for epoch in range(n_epochs):
            loss_train = 0
            loss_valid = 0
            running_loss = 0
            model.train()
            for step in range(1,len(train_loader)+1):
                mfbs, label = next(iter(train_loader))
                mfbs=mfbs.cuda()
                label=label.cuda()
                optimizer.zero_grad()
                prediction=model(mfbs)
                loss=criterion(prediction, label)
                loss.backward()
                optimizer.step()
                loss_train+=loss.item()
            
            # t=time.time()
            # runtime=t-prev_time
            train_loss.append(loss_train/len(train_loader))
            with torch.no_grad():
                for step in range(1,len(val_loader)+1):
                    mfbs, label = next(iter(val_loader))
                    mfbs=mfbs.cuda()
                    label=label.cuda()
                    prediction=model(mfbs)
                    loss=criterion(prediction, label)
                    loss_valid+=loss.item()
                valid_loss.append(loss_train/len(val_loader))

            # if epoch%2==0:
            #     print("epoch=",epoch, "train_loss=",loss_train/len(train_loader),"valid_loss=", loss_valid/len(val_loader),"time=",runtime)
            # prev_time=time.time()

            # if epoch%5==0:
            #     state = {
            #             "epoch": epoch,
            #             "state_dict": model.state_dict(),
            #             }

                # filename = os.path.join("/home/lancelcy/PRIORI/checkpoint",(str(epoch)+".checkpoint"))
                # print(filename)
                # torch.save(state, filename)
            
            #V1_test

        model_test=model.cpu()
        with torch.no_grad():
            Y_out=model_test(torch.from_numpy(X_test)).numpy()
        Y_pred=np.zeros(Y_test.shape[0])
        for i, pred in enumerate(Y_out):
            Y_pred[i]=np.argmax(pred)
        test1_accuracy.append(metrics.accuracy_score(Y_test,Y_pred))
        test1_UAR.append(metrics.recall_score(Y_test,Y_pred,average="macro"))


        with torch.no_grad():
            Y_21_out=model_test(torch.from_numpy(X_R21)).numpy()
            # print(Y_out.shape)
        Y_pred=np.zeros(Y_21_out.shape[0])
        for i, pred in enumerate(Y_21_out):
            Y_pred[i]=np.argmax(pred)
        test2_accuracy.append(metrics.accuracy_score(Y_R21,Y_pred))
        test2_UAR.append(metrics.recall_score(Y_R21,Y_pred,average="macro"))
    np.save("test1_accuracy.npy",test1_accuracy)
    np.save("test1_UAR.npy",test1_UAR)
    np.save("test2_accuracy.npy",test2_accuracy)
    np.save("test2_UAR.npy",test2_UAR)
    print("V1 accuracy",test1_accuracy)
    print("R21 accuracy",test2_accuracy)
    print("V1 UAR",test1_UAR)
    print("R21 UAR",test2_UAR)


if __name__ == "__main__": 
    main()

