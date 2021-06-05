import pickle
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from tqdm import tqdm
from model import *
import argparse

def main():
    parser = argparse.ArgumentParser(description='trial.')
    parser.add_argument('trial', type=str, help='input trial no.')
    t=parser.parse_args().trial
    df=pd.read_csv("/home/lancelcy/PRIORI/metadata/R21_v1.csv")
    idx=np.array(df[df["sub_id"].apply(lambda x: x==2280001 or x==5819001 or x== 5865001 or x==5998001)].iloc[:,0])
    with open("/home/lancelcy/PRIORI/data/X_R21.pk","rb") as f:
    	X=np.take(pickle.load(f),idx,axis=0)
    with open("/home/lancelcy/PRIORI/data/Y_R21.pk","rb") as f:
    	Y=np.take(pickle.load(f),idx,axis=0)
    print(X.shape)
#     X=np.split(X_R21,[816,933,1123,1169,2477,2843,3838,3957])
#     Y=np.split(Y_R21,[816,933,1123,1169,2477,2843,3838,3957])
#     accuracy=np.zeros((9,30))
    UAR=np.zeros(30)
    for j in tqdm(range(30)):
        dic=torch.load(f"/home/lancelcy/PRIORI/checkpt/CNN_models/trial1/{j}.checkpoint",map_location="cpu")
        model= Net()
        model.load_state_dict(dic)
        # subject_no=0
        # for x,y in zip(X,Y):
            # print(x.shape)
        with torch.no_grad():
            Y_out=model(torch.from_numpy(X)).numpy()
                # print(Y_out.shape)
            Y_pred=np.zeros(Y.shape[0])
            for i, pred in enumerate(Y_out):
                Y_pred[i]=np.argmax(pred)
        UAR[j]=metrics.recall_score(Y,Y_pred,average="macro")
        print(metrics.recall_score(Y,Y_pred,average="macro"))

            # subject_no+=1
    # np.save(f"/home/lancelcy/PRIORI/test_result/baseline/trial{t}/accuracy.npy",accuracy)
    np.save(f"/home/lancelcy/PRIORI/test_result/baseline/trial{t}/UAR_v1.npy",UAR)



if __name__ == "__main__": 

    main()
