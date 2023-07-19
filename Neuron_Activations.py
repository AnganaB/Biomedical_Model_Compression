import pandas as pd
import numpy as np
import csv
import argparse
import math
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import Dataset, DataLoader
from torch import cuda
from seqeval.metrics import classification_report
# from config import Config as config
import os
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
from transformers.modeling_outputs import TokenClassifierOutput
import clf_distill_loss_functions
from clf_distill_loss_functions import *
import warnings
import os, sys
from matplotlib import pyplot as plt
%matplotlib inline
import pickle
import gzip

sys.path.append("..")


from transformers import AutoTokenizer, AutoConfig, AutoModelForTokenClassification, AutoModel, BertPreTrainedModel

input_path = '/raid/data/kay/mounted/'
output_path = 'resources'
log_soft = F.log_softmax
tokenizer_dir = "/raid/data/kay/mounted/tokenizer"
model_dir = "/raid/data/kay/mounted/model"
config_dir = "/raid/data/kay/mounted/config"

print(torch.version.cuda)
MAX_LEN = 310# suitable for all datasets
MAX_GRAD_NORM = 10
BATCH_SIZE = 16
LEARNING_RATE = 3e-5


def _plot_helper(arr, xlabel, ylabel):
    plt.plot(arr, lw=2.0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()

def neuron_act(train_dataloader, model):
    #Forward hook 
    
    h1 = model.bert.encoder.layer[11].output.dropout.register_forward_hook(get_features('feats'))
    

        # neuron activations 
    FEATS = []

    # placeholder for batch features
    features = {}
    k = 0
    # loop through batches
    for batch in train_dataloader:

        device = "cpu"
        input_ids = batch['ids'].to(device, dtype=torch.long)
        mask = batch['mask'].to(device, dtype=torch.long)
        targets = batch['targets'].to(device, dtype=torch.long)

        preds = model(input_ids=input_ids, attention_mask=mask, labels=targets)


        # add feats to lists
        FEATS.append(features['feats'].cpu().numpy())
     
    FEATS = np.concatenate(FEATS)
    feats_enc = FEATS_encoder_0[:,309,:] # taking the last or can take the average too 
    
    h1.remove() # close the hook 
    
    return feats_enc  


def main():
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    model = MyModel.from_pretrained(model_dir,num_labels = len(label2id),loss_fn = loss_fn)
    device = 'cpu'
    loss_fn = clf_distill_loss_functions.Plain()
    
    # dataloader - train needed for understanding the network 
    train_data = {'Sentence':train_sent, 'Labels':train_label}
    train_data = pd.DataFrame(train_data)
    devel_data = {'Sentence':devel_sent, 'Labels':devel_label}
    devel_data = pd.DataFrame(devel_data)

    train_params = {'batch_size': BATCH_SIZE,
                    'shuffle': False,
                    }

    devel_params = {'batch_size': BATCH_SIZE,
                    'shuffle': True,
                    }

    train_dataset = dataset(train_data, tokenizer, MAX_LEN, label2id, id2label)
    train_dataloader = DataLoader(train_dataset, **train_params)
    devel_dataset = dataset(devel_data, tokenizer, MAX_LEN, label2id, id2label)
    devel_dataloader = DataLoader(devel_dataset, **devel_params)


    #load the model 
    model.load_state_dict(torch.load("/raid/data/kay/mounted/BestModel__ncbi/pytorch_model.bin",  map_location="cuda:2"))

    print(model)
    
    feats = neuron_act(train_dataloader, model) # neuron activations from the pooler layer before dropout and classifier 
    act = pd.DataFrame(feats)
    act.to_csv("activations_pool_train.csv")

    act['highest'] = act.idxmax(axis=1).tolist()
    print(act.value_counts()) # print the top activated neurons for the train set 
    act.to_csv("activation_pool_train_highest.csv")

    
# %%time
if __name__ == '__main__':
    main()
    
    