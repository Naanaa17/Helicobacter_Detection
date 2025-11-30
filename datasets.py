import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
#import torchvision

import numpy as np
from random import shuffle
import random


# =============================================================================
# Triplet dataset 
# =============================================================================
# X needs to have structure [NSamp,...] or be a list of NSamp entries
class TripletDataset():
    def __init__(self, images,labels,transform=None):

        self.images = images
        self.labels = labels
        self.index = np.arange(labels.shape[0]) 
        self.transform=transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, item):

        ###### Anchor Image #######
        anchor_img = torch.from_numpy(self.images[item]).float()
        anchor_label = self.labels[item]
        
        positive_list = self.index[self.index!=item][self.labels[self.index!=item]==anchor_label]
        positive_item = random.choice(positive_list)
        positive_img = torch.from_numpy(self.images[positive_item]).float()
        
        #positive_img = self.images[positive_item].reshape(28, 28, 1)
        negative_list = self.index[self.index!=item][self.labels[self.index!=item]!=anchor_label]
        negative_item = random.choice(negative_list)
        negative_img = torch.from_numpy(self.images[negative_item]).float()
            #negative_img = self.images[negative_item].reshape(28, 28, 1)
        if self.transform!=None:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)                   
            negative_img = self.transform(negative_img)
                
        return anchor_img, positive_img, negative_img, torch.from_numpy(np.array(anchor_label))    
# =============================================================================
# Standard dataset (Single Objective)
# =============================================================================
# X needs to have structure [NSamp,...] or be a list of NSamp entries
class Standard_Dataset(data.Dataset):
    def __init__(self, X, Y, transformation=None):
        super().__init__()
        self.X = X
        self.y = Y
        self.transformation = transformation
 
    def __len__(self):
        
        return len(self.X)

    def __getitem__(self, idx):
        
        return torch.from_numpy(self.X[idx]).float(), torch.from_numpy(np.array(self.y[idx]))


# =============================================================================
#  Pairs of Random Instances
# =============================================================================
class Triple_Dataset(data.Dataset):
    """ Radiomics Dataset handle the 1D extracted features. """
    
    def __init__(self, X, y, idxTriple):
        super().__init__()
        self.X = X
        self.y = y
        self.idxTriple=idxTriple
        


    
    def __len__(self):
            return len(self.X)
    

        
    def __getitem__(self, idx):  
        
        if idx % 2 == 0: # selects samples from the same class
        #    shuffle(self.idx_Same)
          # random_index1 = self.idx_Same[0] 
            random_index1 = np.random.choice(len(self.idx_Same), 1)[0]
            random_index1 =   self.idx_Same[random_index1]       
        else: # selects samples from different classes
            # shuffle(self.idx_Diff)
            # random_index1 = self.idx_Diff[0] 
           # random_index1 = np.random.choice(len(idx_Diff), 1)[0]
            random_index1 = np.random.choice(len(self.idx_Diff), 1)[0]
            random_index1 =   self.idx_Diff[random_index1]  
            
        return torch.from_numpy(self.X[random_index1]).float(), \
            torch.from_numpy(np.array(self.y[random_index1])).long()

# =============================================================================
#  Pairs of Random Instances
# =============================================================================
class Paired_Dataset(data.Dataset):
    """ Radiomics Dataset handle the 1D extracted features. """
    
    def __init__(self, X, y, transform=None):
        super().__init__()
        self.X = X
        self.y = y
        self.idx_Same=np.nonzero(self.y==0)[0]
        self.idx_Diff=np.nonzero(self.y==1)[0]
        # _, self.counts = np.unique(self.y, return_counts=True)
        
        self.transform = transform

    
    def __len__(self):
            return len(self.X)
    

        
    def __getitem__(self, idx):  
        sample1 = None
        sample2 = None
        label = None
        
        if idx % 2 == 0: # selects samples from the same class
        #    shuffle(self.idx_Same)
          # random_index1 = self.idx_Same[0] 
            random_index1 = np.random.choice(len(self.idx_Same), 1)[0]
            random_index1 =   self.idx_Same[random_index1]       
        else: # selects samples from different classes
            # shuffle(self.idx_Diff)
            # random_index1 = self.idx_Diff[0] 
           # random_index1 = np.random.choice(len(idx_Diff), 1)[0]
            random_index1 = np.random.choice(len(self.idx_Diff), 1)[0]
            random_index1 =   self.idx_Diff[random_index1]  
            
        return torch.from_numpy(self.X[random_index1]).float(), \
            torch.from_numpy(np.array(self.y[random_index1])).long()
               
class Paired_Dataset_Guille(data.Dataset):
    """ Radiomics Dataset handle the 1D extracted features. """
    
    def __init__(self, X, y, transform=None):
        super().__init__()
        self.X = X
        self.y = y
        self.idxs=np.arange(y.shape[0])
        # _, self.counts = np.unique(self.y, return_counts=True)
        
        self.transform = transform

    
    def __len__(self):
            return len(self.X)
    

    def __getitem__(self, idx):  
        sample1 = None
        sample2 = None
        label = None
        
        shuffle(self.idxs)
        
        if idx % 2 == 0: # selects samples from the same class
            random_index1 = np.random.choice(len(self.y), 1)[0]
            random_class = self.y[random_index1]
            
            indexes = np.where(self.y == random_class)[0]  # same class
            random_index2 = np.random.choice(indexes)
            
            assert self.y[random_index1] == self.y[random_index2]
            
            sample1 = self.X[random_index1]
            sample2 = self.X[random_index2]
            
            gt = 1
            
        else: # selects samples from different classes
            random_index1 = np.random.choice(len(self.y), 1)[0]
            random_class1 = self.y[random_index1]
            
            indexes = np.where(self.y != random_class1)[0] # different class
            random_index2 = np.random.choice(indexes)
            
            assert self.y[random_index1] != self.y[random_index2]
            
            sample1 = self.X[random_index1]
            sample2 = self.X[random_index2]
            
            gt = 0

        return (torch.from_numpy(sample1.astype(np.float32)).float(), \
               torch.from_numpy(sample2.astype(np.float32)).float()), \
               torch.from_numpy(np.array(gt)).long()
               
       
        
        
# =============================================================================
# Transformations
# =============================================================================

class IdentitySeries(object):
    """
    x is numpy dtype
    """
    def __call__(self, x):
        return x  # [n_feats, time_steps]

class ReverseSeries(object):
    """
    x is numpy dtype
    """
    def __call__(self, x):
        x = np.ascontiguousarray(x[:, ::-1])
        return x # [n_feats, time_steps]

class ToTensorSeries(object):
    """
    x is numpy dtype
    """
    def __call__(self, x):

        return torch.from_numpy(x) # [n_feats, time_steps]

"""
TFG:Valorar si val la pena posar això en un .py a part
    
"""
# =============================================================================
# Data Sets Creation
# =============================================================================

def class_sample_count(labels):
    u,indices=np.unique(np.sort(labels.flatten()),return_index=True)
    nword=np.diff(np.array(list(indices)+[len(labels.flatten())]))
    return nword

def classes_weight(y_train):
    
    
    # computing a weight per class/sample only is util when
    # you are dealing with unbalanced data, however, it does
    # not matter with balanced dataset

    sample_counts = np.array(class_sample_count(y_train))
    #sample_counts = np.array(class_sample_count(y_train))
    classes_weight=1./sample_counts
    classes_weight=classes_weight/np.sum(classes_weight)
    classes_weight=torch.tensor(classes_weight, dtype=torch.float).cuda()
    
        
    return classes_weight


def classes_weight_binary(y_train):
    
    
    # computing a weight per class/sample only is util when
    # you are dealing with unbalanced data, however, it does
    # not matter with balanced dataset

    sample_counts = np.array(np.sum(y_train,axis=0))
    # avoid division by zero, and therefore, inf values in the result
 #   sample_counts[sample_counts==0] = 1
    classes_weight=1./sample_counts
    classes_weight[classes_weight==np.inf]=0
    classes_weight=classes_weight/np.sum(classes_weight)
    classes_weight=torch.tensor(classes_weight, dtype=torch.float).cuda()
    
        
    return classes_weight

def create_Paired_dataloader(x_test, y_test, transf=False, batch_size=128, shuffle=True):

#
#    if y_test.shape[0] < batch_size:
#        batch_size = y_test.shape[0]

    test_dataset = Paired_Dataset(x_test, y_test)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                       batch_size=batch_size,
                                                       shuffle=shuffle)
    return test_dataloader


def create_dataloader(x_test, y_test, transf=False, batch_size=128, shuffle=True):

#
#    if y_test.shape[0] < batch_size:
#        batch_size = y_test.shape[0]

    test_dataset = Standard_Dataset(x_test, y_test, transf)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                       batch_size=batch_size,
                                                       shuffle=shuffle)
    return test_dataloader


def create_dataloader_balanced(x_train, y_train, transf=False, batch_size=128, shuffle=False):
#    if y_train.ndim > 1:
#        print('Data was no properly encoded. Error in train_dataloader!')
#        return

    sample_counts = class_sample_count(list(y_train))
    classes_weight = 1. / torch.tensor(sample_counts, dtype=torch.float)
    samples_weight = torch.tensor([classes_weight[w] for w in y_train])

    # samples_weight=classes_weight(y_train)
    # traind dataloader
    train_dataset = Standard_Dataset(x_train, y_train, transf)
    

    # pytorch function for sampling batch based on weights or probabilities for each
    # element. To obtain a relative balaced batch, it uses replacement by default
    sampler = torch.utils.data.WeightedRandomSampler(weights=samples_weight,
                                                     num_samples=len(samples_weight),
                                                     replacement=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                   shuffle=shuffle, sampler=sampler)
    return train_dataloader

