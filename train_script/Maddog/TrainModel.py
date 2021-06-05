import sys, os
import numpy as np
from DataReader import DataReader
import argparse
from torch.utils.data.sampler import BatchSampler
from torch.utils.data import DataLoader
from Samplers import BatchReplace, SubsetSampler
from Maddog import Maddog
from IPython import embed

# Calculate the unweighted average recall (UAR)
def calcUar(actual, pred):
    unVals = np.unique(actual)
    sumAcc = 0.0
    for unVal in unVals:
        actPred = pred[actual==unVal]
        sumAcc += float(np.sum(actPred==unVal))/len(actPred)
    sumAcc = sumAcc / float(len(unVals))
    return sumAcc

def run(args):
    # Load data
    data = DataReader(args.input)
    
    # Determine data for validation
    data.loadFeatures = False
    isVal = np.array([x['Validation'] for x in data])
    isTrn = np.logical_not(isVal)
    hasEmo = np.array([not np.any(np.isnan(x['Emotion'])) for x in data])
    isTrnEmo = np.logical_and(isTrn, hasEmo)
    
    # Create samplers
    bsVal = BatchSampler(SubsetSampler(np.where(isVal)[0]), args.batchSize, False)
    dlVal = DataLoader(data, collate_fn=data.collate, batch_sampler=bsVal)
    bsTrn = BatchReplace(np.where(isTrn)[0], args.batchSize)
    dlTrn = DataLoader(data, collate_fn=data.collate, batch_sampler=bsTrn)
    itTrn = iter(dlTrn)
    bsTrnEmo = BatchReplace(np.where(isTrnEmo)[0], args.batchSize)
    dlTrnEmo = DataLoader(data, collate_fn=data.collate, batch_sampler=bsTrnEmo)
    itTrnEmo = iter(dlTrnEmo)
    
    # Get output weights using train and val data
    allEmo = np.stack([x['Emotion'] for x in data])
    allDs  = np.stack([x['Dataset'] for x in data])
    wEmo = np.nansum(allEmo,axis=0)
    wDs = np.nansum(allDs,axis=0)
    wEmo = np.sum(wEmo)/(wEmo*len(wEmo))
    wDs = (np.sum(wDs)-wDs)/wDs
    data.loadFeatures = True
    
    # Setup model
    model = Maddog(data.featLen, wDs, wEmo, args)
    
    # Get validation ground truth
    valActual = np.concatenate([x['Emotion'] for x in dlVal])
    valActual = np.argmax(valActual, axis=-1)
    
    # Loop through all epochs
    bestUar = None
    for ep in range(args.maxEpochs):
        # Train for one epoch
        model.Fit(itTrn, itTrnEmo)

        # Predict VAL
        valPred = np.argmax(model.Predict(dlVal), axis=-1)
        valUar = calcUar(valActual, valPred)
        
        # Print UAR
        if args.verbose:
            print('Epoch', ep, '  UAR:', valUar)

        # Check for best val UAR
        if bestUar is None or valUar > bestUar:
            bestUar = valUar
            model.Save(args.model)

def main():
    # Parse commandline arguments
    parser = argparse.ArgumentParser(description='Train and save a MADDoG model.')
    parser.add_argument('input', type=str, help='The path to the input data csv file')
    parser.add_argument('model', type=str, help='The path to where to save the model')
    parser.add_argument('-b', '--batchSize', type=int, default=32, help='The batch size during training (32)')
    parser.add_argument('-r', '--repLen', type=int, default=128, help='The size of the mid-level representation (128)')
    parser.add_argument('-g', '--gFactor', type=float, default=0.1, help='The weighting factor for the dataset output loss when training the generator (0.1)')
    parser.add_argument('-d', '--dRepeats', type=int, default=5, help='The amount of repeat batches to train the critic before the generator (5)')
    parser.add_argument('-l', '--learningRate', type=float, default=0.0001, help='The learning rate during training (0.0001)')
    parser.add_argument('-e', '--maxEpochs', type=int, default=30, help='The maximum number of epochs (30)')
    parser.add_argument('-v', '--verbose', action='store_true', help='A flag to print validation UARs (false)')
    args = parser.parse_args()
    run(args)
    
if __name__ == "__main__": main()
