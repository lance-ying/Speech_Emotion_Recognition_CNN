import pickle
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from model import * 
from tqdm import tqdm
import argparse

def main():
    # parser = argparse.ArgumentParser(description='trial.')
    # parser.add_argument('trial', type=str, help='input trial no.')
    # t=parser.parse_args().trial
    print("loading data")
    X_V1=np.load("/home/lancelcy/PRIORI/data/X_V1.npy")
    Y_V1=np.load("/home/lancelcy/PRIORI/data/Y_V1.npy")
    df=pd.read_csv("/home/lancelcy/PRIORI/metadata/split_data.csv")
    Tes_idx=np.array(df[df["trial1"].apply(lambda x:x==2)].index)
    X=np.take(X_V1,Tes_idx,axis=0)
    Y=np.take(Y_V1,Tes_idx,axis=0)
    print(X.shape)

    accuracy=np.zeros(30)
    UAR=np.zeros(30)
    count=0
    for j in tqdm(range(30)):
        dic=torch.load(f"/home/lancelcy/PRIORI/checkpt/R21_CNN/{j}.checkpoint",map_location="cpu")
        model = Net()
        model.load_state_dict(dic)

        with torch.no_grad():
            Y_out=model(torch.from_numpy(X)).numpy()
            Y_pred=np.zeros(Y.shape[0])
            for i, pred in enumerate(Y_out):
                Y_pred[i]=np.argmax(pred)

        accuracy[count]=metrics.accuracy_score(Y,Y_pred)
        UAR[count]=metrics.recall_score(Y,Y_pred,average="macro")
        count+=1
    np.save(f"/home/lancelcy/PRIORI/test_result/baseline/R21/accuracy_V1.npy",accuracy)
    np.save(f"/home/lancelcy/PRIORI/test_result/baseline/R21/UAR_V1.npy",UAR)

if __name__ == "__main__": 

    main()
