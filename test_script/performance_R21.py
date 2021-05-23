import pickle
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import * 
import argparse

def main():
    parser = argparse.ArgumentParser(description='trial.')
    parser.add_argument('trial', type=str, help='input trial no.')
    t=parser.parse_args().trial
    with open("/home/lancelcy/PRIORI/data/X_R21.pk","rb") as f:
    	X_R21=pickle.load(f)
    with open("/home/lancelcy/PRIORI/data/Y_R21.pk","rb") as f:
    	Y_R21=pickle.load(f)
    X=np.split(X_R21,[816,933,1123,1169,2477,2843,3838,3957])
    Y=np.split(Y_R21,[816,933,1123,1169,2477,2843,3838,3957])
    accuracy=np.zeros((9,30))
    UAR=np.zeros((9,30))
    for j in tqdm(range(30)):
        dic=torch.load(f"/home/lancelcy/PRIORI/CNN_models/{j}.checkpoint",map_location="cpu")
        model = Net()
        model.load_state_dict(dic)
        subject_no=0
        for x,y in zip(X,Y):
            print(x.shape)
            with torch.no_grad():
                Y_out=model(torch.from_numpy(x)).numpy()
                # print(Y_out.shape)
            Y_pred=np.zeros(y.shape[0])
            for i, pred in enumerate(Y_out):
                Y_pred[i]=np.argmax(pred)

            accuracy[subject_no,j]=metrics.accuracy_score(y,Y_pred)
            UAR[subject_no,j]=metrics.recall_score(y,Y_pred,average="macro")
            subject_no+=1
    np.save(f"/home/lancelcy/PRIORI/test_result/baseline/trial{t}/accuracy.npy",accuracy)
    np.save(f"/home/lancelcy/PRIORI/test_result/baseline/trial{t}/UAR.npy",UAR)



if __name__ == "__main__": 

    main()
