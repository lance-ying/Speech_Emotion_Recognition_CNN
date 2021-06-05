import pandas as pd
import numpy as np
from sklearn import metrics

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from model import * 
from tqdm import tqdm
import argparse

os.chdir("/home/lancelcy/PRIORI")

def main():
    # parser = argparse.ArgumentParser(description='trial.')
    # parser.add_argument('trial', type=str, help='input trial no.')
    # t=parser.parse_args().trial
    print("loading data")

    X=np.load("/home/lancelcy/PRIORI/data/X_R21.npy")
    Y=np.load("/home/lancelcy/PRIORI/data/Y_R21.npy")
    
    df=pd.read_csv("/home/lancelcy/PRIORI/metadata/R21_v1.csv")
    Tr_idx=np.array(df[df["trial1"].apply(lambda x:x==0)].index)
    Val_idx=np.array(df[df["trial1"].apply(lambda x:x==1)].index)
    Tes_idx=np.array(df[df["trial1"].apply(lambda x:x==2)].index)
    X_train=np.take(X,Tr_idx,axis=0)
    X_val=np.take(X,Val_idx,axis=0)
    Y_train=np.take(Y,Tr_idx,axis=0)
    Y_val=np.take(Y,Val_idx,axis=0)
    X_test=np.take(X,Tes_idx,axis=0)
    Y_test=np.take(Y,Tes_idx,axis=0)
    
 
    print("data loaded")

    train_data=TensorDataset(torch.FloatTensor(X_train),torch.LongTensor(Y_train))
    val_data=TensorDataset(torch.FloatTensor(X_val),torch.LongTensor(Y_val))
    train_loader=DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader=DataLoader(val_data, batch_size=64, shuffle=True)
    UAR=[]
    # training the model
    for i in tqdm(range(30)):
        #initializing
        
        model = Net()
        best_model=None
        best_loss=np.inf
        epoch=0
        patience=5
        cur_patience=patience

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001,weight_decay=1e-6)
        if torch.cuda.is_available():
            model = model.cuda(0)
            criterion = criterion.cuda(0)

        #training
        while cur_patience!=0:
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
            print("epoch ", epoch, "train_loss=",new_train_loss,"val_loss=",new_val_loss)

            if new_val_loss<=best_loss:
                # print(new_val_loss)
                best_model=model.state_dict()
                # print(best_model)
                best_loss=new_val_loss
                cur_patience=patience
            else:
                cur_patience-=1

            epoch+=1
        with torch.no_grad():
            model= Net()
            model.load_state_dict(best_model)
            Y_out=model(torch.from_numpy(X_test)).numpy()
        Y_pred=np.zeros(Y_test.shape[0])
        for j, pred in enumerate(Y_out):
            Y_pred[j]=np.argmax(pred)
        perf=metrics.recall_score(Y_test,Y_pred, average="macro")
        print(perf)
        UAR.append(perf)
        # print(i)
        filename=os.path.join(f"/home/lancelcy/PRIORI/checkpt/R21_CNN",f"{i}.checkpoint")
        # print(filename)
        with torch.no_grad():
            torch.save(best_model, filename)
    
    np.save("/home/lancelcy/PRIORI/test_result/R21_test.npy",np.array(UAR))
        

if __name__ == "__main__": 

    main()

            
        