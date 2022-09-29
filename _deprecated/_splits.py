import argparse
import numpy as np
import os
import random
import torch 
from torch.utils.data import random_split

def dataset_split(dataset, setting): 
    if setting in ['EC', 'GO-BP', 'GO-MF', 'GO-CC']:
        train_dataset, validation_dataset, test_dataset = multi_cutoff_split(dataset) 
    elif setting in ['FC-Fold', 'FC-Super', 'FC-Fam']: 
        train_dataset, validation_dataset, test_dataset = fold_classification_split(dataset, setting)
    elif setting in ['Reaction']:
        train_dataset, validation_dataset, test_dataset = reaction_split(dataset)
    return train_dataset, validation_dataset, test_dataset

def multi_cutoff_split(dataset):
    """
    Gligorijevic et al., 2021
    Wang et al., 2022
    the test set only contains PDB chains with sequence identity no more than 95% to the training set
    """
    train_dataset, validation_dataset, test_dataset = None, None, None
    return train_dataset, validation_dataset, test_dataset

def fold_classification_split(dataset, setting):
    """
    Hou et al., 2018
    Fold: proteins from the same superfamily are unseen during training
    Superfamily: proteins from the same family are not present during training
    Family: proteins from the same family are present during training
    """
    train_dataset, validation_dataset, test_dataset = None, None, None
    return train_dataset, validation_dataset, test_dataset

def reaction_split(dataset):
    """
    Hermosilla et al., 2021
    proteins have less than 50% sequence similarity in-between splits
    """
    train_dataset, validation_dataset, test_dataset = None, None, None
    return train_dataset, validation_dataset, test_dataset

