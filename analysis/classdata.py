import numpy as np
import pandas as pd

class Data:
    def __init__(self, metadata, pixels, fc6):
        self.metadata = metadata
        self.pixels = pixels
        self.fc6 = fc6
    def filter_out(self, kinds = {'invalid' : True,
                                  'outlier' : True}):
        '''
        input: 
            frame = pandas dataframe
            remove = boolean flag, if True will remove flagged data
            kind = str, can be either 'invalid' or 'outlier'
        output:
            pandas dataframe (subset of frame)
            pixel-level features (subsetted)
            fc6 features (subsetted)
        '''
        kind2col = {'invalid': 'isInvalid', 'outlier': 'isOutlier'} # mapping between keyword arg and column name in frame
        for kind in kinds:
            self.metadata = self.metadata[self.metadata[kind2col[kind]] != True] if kinds[kind] else self.metadata
        self.pixels = self.pixels[self.metadata.index]
        self.fc6 = self.fc6[self.metadata.index]
        self.metadata = self.metadata.reset_index(drop=True)
    def preprocess(self, mean=True, std=False):
        '''
        input:
            mean = boolean, if true, will normalize features to mean = 0
            std  = boolean, if true, will normalize features to std = 1
        output:
            updates feature parameters to be normalized/standardized
        '''
        stds = self.pixels.std(axis = 0)
        stds[stds == 0.] = 1
        means = self.pixels.mean(axis = 0)
        self.pixels = self.pixels - means if mean else self.pixels
        self.pixels = self.pixels / stds if std else self.pixels
        
        stds = self.fc6.std(axis = 0)
        stds[stds == 0.] = 1
        means = self.fc6.mean(axis = 0)
        self.fc6 = self.fc6 - means if mean else self.fc6
        self.fc6 = self.fc6 / stds if std else self.fc6
        
        if std == True:
            assert round(np.sum(self.pixels.std(axis=0)) + np.sum((self.pixels.std(axis=0) == 0.))) == len(self.pixels[0])
            assert round(np.sum(self.fc6.std(axis=0)) + np.sum((self.fc6.std(axis=0) == 0.))) == len(self.fc6[0])
        if mean == True:
            assert round(np.sum(self.pixels.mean(axis=0))) == 0
            assert round(np.sum(self.fc6.mean(axis=0))) == 0