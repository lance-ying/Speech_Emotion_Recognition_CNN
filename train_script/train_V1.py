import os
from numpy.core.numeric import Infinity
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from model import * 
from tqdm import tqdm
import argparse

def main():
    parser = argparse.ArgumentParser(description='trial.')
    parser.add_argument('trial', type=str, help='input trial no.')
    t=parser.parse_args().trial

    print("loading data")

    X_V1=np.load("/home/lancelcy/PRIORI/data/X_V1.npy")
    Y_V1=np.load("/home/lancelcy/PRIORI/data/Y_V1.npy")
    df=pd.read_csv("/home/lancelcy/PRIORI/metadata/split_data.csv")
    Tr_idx=np.array(df[df[f"trial{t}"].apply(lambda x:x==0)].index)
    Val_idx=np.array(df[df[f"trial{t}"].apply(lambda x:x==1)].index)
    Tes_idx=np.array(df[df[f"trial{t}"].apply(lambda x:x==2)].index)
    X_train=np.take(X_V1,Tr_idx,axis=0)
    X_val=np.take(X_V1,Val_idx,axis=0)
    Y_train=np.take(Y_V1,Tr_idx,axis=0)
    Y_val=np.take(Y_V1,Val_idx,axis=0)
 
    print("data loaded")

    train_data=TensorDataset(torch.FloatTensor(X_train),torch.LongTensor(Y_train))
    val_data=TensorDataset(torch.FloatTensor(X_val),torch.LongTensor(Y_val))
    train_loader=DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader=DataLoader(val_data, batch_size=64, shuffle=True)
    
    patience=5

    # training the model
    for i in tqdm(range(30)):

        #initializing
        model = Net()
        best_model=None
        best_loss=np.float("inf")
        epoch=0

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001,weight_decay=1e-6)
        loss_min = np.inf
        if torch.cuda.is_available():
            model = model.cuda(0)
            criterion = criterion.cuda(0)

        #training
        while patience!=0:
            loss_train = 0
            loss_valid = 0
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
            
            new_train_loss=loss_train/len(train_loader)

            with torch.no_grad():
                for step in range(1,len(val_loader)+1):
                    mfbs, label = next(iter(val_loader))
                    mfbs=mfbs.cuda()
                    label=label.cuda()
                    prediction=model(mfbs)
                    loss=criterion(prediction, label)
                    loss_valid+=loss.item()
            new_val_loss=loss_valid/len(val_loader)
            print("epoch ", epoch, "train_loss=",new_train_loss,"val_loss=",new_val_loss,"\n")

            if new_val_loss<=best_loss:
                best_model=model
                best_loss=new_val_loss
                patience=5
            else:
                patience-=1

            epoch+=1
        filename=os.path.join(os.cwd(),f"/checkpt/CNN_models/trial{t}",f"{j}.checkpoint")
        torch.save(best_model.state_dict(), filename)
        

if __name__ == "__main__": 

    main()

            
        