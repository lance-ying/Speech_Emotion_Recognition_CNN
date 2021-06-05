from torch.utils.data import Dataset, DataLoader
import csv
import numpy as np
import copy
  
# Pad features with zeros to a certain length
def zeroPadToLength(data, length):
    padAm = length - data.shape[-1]
    if padAm == 0:
        return data
    else:
        return np.pad(data, ((0,0),(0,padAm)), 'constant')
  
class DataReader(Dataset):
    def __init__(self, path):
        # Parse metadata file
        self.meta = []
        self.nDatasets = 0
        self.nEmotions = 0
        self.featLen = 0
        self.loadFeatures = True
        hasVal = False
        with open(path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                entry = {}
                entry['FeaturePath'] = row['features']
                if 'dataset' in row:
                    entry['Dataset'] = int(row['dataset'])
                    self.nDatasets = max(self.nDatasets, entry['Dataset']+1)
                if 'emotion' in row:
                    if row['emotion'] == '':
                        entry['Emotion'] = None
                    else:
                        entry['Emotion'] = np.array([float(x) for x in row['emotion'].split(';')])
                        self.nEmotions = len(entry['Emotion'])
                if 'validation' in row:
                    if row['validation'] == '':
                        entry['Validation'] = False
                    else:
                        entry['Validation'] = int(row['validation'])!=0
                    hasVal = True
                self.meta.append(entry)
        
        # If no validation set given and training, use random 1/5 split
        if not hasVal and self.nEmotions > 0:
            for sample in self.meta:
                sample['Validation'] = False
                if sample['Emotion'] is not None:
                    sample['Validation'] = (np.random.uniform() < 0.8)
                    
        # Change None emotion to NaNs
        if self.nEmotions > 0:
            for sample in self.meta:
                if sample['Emotion'] is None:
                    sample['Emotion'] = np.full((self.nEmotions,), np.nan)
                    
        # Get feature length
        self.featLen = np.load(self.meta[0]['FeaturePath']).T.shape[0]
        
        # Convert Dataset to OH
        if self.nDatasets > 0:
            for sample in self.meta:
                curDs = sample['Dataset']
                sample['Dataset'] = np.zeros((self.nDatasets,))
                sample['Dataset'][curDs] = 1

    def __len__(self):
        return len(self.meta)
        
    def __getitem__(self, idx):
        sample = copy.deepcopy(self.meta[idx])
        if self.loadFeatures:
            sample['Features'] = np.load(sample['FeaturePath']).T
        return sample
        
    @staticmethod    
    def collate(samples):
        batch = {}
        names = ['Features', 'Emotion', 'Dataset']
        for name in names:
            if name in samples[0]:
                if name == 'Features':
                    lengths = [sample['Features'].shape[-1] for sample in samples]
                    maxLen = np.max(lengths)
                    features = [zeroPadToLength(sample['Features'], maxLen) for sample in samples]
                    batch['Features'] = np.stack(features)
                else:
                    batch[name] = np.stack([x[name] for x in samples])
        return batch