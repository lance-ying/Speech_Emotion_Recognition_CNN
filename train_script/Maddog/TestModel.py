import sys, os
import numpy as np
from DataReader import DataReader
import argparse
from torch.utils.data.sampler import BatchSampler, SequentialSampler
from torch.utils.data import DataLoader
from Maddog import Maddog
import csv

def run(args):
    # Load data
    data = DataReader(args.input)
        
    # Create sampler
    bsTst = BatchSampler(SequentialSampler(data), args.batchSize, False)
    dlTst = DataLoader(data, collate_fn=data.collate, batch_sampler=bsTst)
    
    # Setup model
    model = Maddog(ptPath=args.model)
    model.Load(args.model)
    
    # Get predictions
    preds = model.Predict(dlTst)    
    data.loadFeatures = False
    with open(args.output, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['features', 'emotion'])
        for i in range(len(data)):
            row = []
            row.append(data[i]['FeaturePath'])
            row.append(';'.join([str(x) for x in preds[i]]))
            writer.writerow(row)
    
def main():
    # Parse commandline arguments
    parser = argparse.ArgumentParser(description='Test a previously trained MADDoG model and output emotion predictions.')
    parser.add_argument('input', type=str, help='The path to the input data csv file')
    parser.add_argument('model', type=str, help='The path to the model directory')
    parser.add_argument('output', type=str, help='The path to where to save the emotion predictions')
    parser.add_argument('-b', '--batchSize', type=int, default=32, help='The batch size during testing (32)')
    args = parser.parse_args()
    run(args)
    
if __name__ == "__main__": main()
