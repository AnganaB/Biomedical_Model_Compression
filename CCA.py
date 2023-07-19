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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

num_cca_trials = 5

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


def positivedef_matrix_sqrt(array):
    w, v = np.linalg.eigh(array)
    #  A - np.dot(v, np.dot(np.diag(w), v.T))
    wsqrt = np.sqrt(w)
    sqrtarray = np.dot(v, np.dot(np.diag(wsqrt), np.conj(v).T))
    return sqrtarray


def remove_small(sigma_xx, sigma_xy, sigma_yx, sigma_yy, epsilon):
    x_diag = np.abs(np.diagonal(sigma_xx))
    y_diag = np.abs(np.diagonal(sigma_yy))
    x_idxs = (x_diag >= epsilon)
    y_idxs = (y_diag >= epsilon)

    sigma_xx_crop = sigma_xx[x_idxs][:, x_idxs]
    sigma_xy_crop = sigma_xy[x_idxs][:, y_idxs]
    sigma_yx_crop = sigma_yx[y_idxs][:, x_idxs]
    sigma_yy_crop = sigma_yy[y_idxs][:, y_idxs]

    return (sigma_xx_crop, sigma_xy_crop, sigma_yx_crop, sigma_yy_crop,
          x_idxs, y_idxs)


def compute_ccas(sigma_xx, sigma_xy, sigma_yx, sigma_yy, epsilon, verbose=True):
    (sigma_xx, sigma_xy, sigma_yx, sigma_yy,
    x_idxs, y_idxs) = remove_small(sigma_xx, sigma_xy, sigma_yx, sigma_yy, epsilon)

    numx = sigma_xx.shape[0]
    numy = sigma_yy.shape[0]

    if numx == 0 or numy == 0:
    return ([0, 0, 0], [0, 0, 0], np.zeros_like(sigma_xx),
            np.zeros_like(sigma_yy), x_idxs, y_idxs)

    if verbose:
    print("adding eps to diagonal and taking inverse")
    sigma_xx += epsilon * np.eye(numx)
    sigma_yy += epsilon * np.eye(numy)
    inv_xx = np.linalg.pinv(sigma_xx)
    inv_yy = np.linalg.pinv(sigma_yy)

    if verbose:
    print("taking square root")
    invsqrt_xx = positivedef_matrix_sqrt(inv_xx)
    invsqrt_yy = positivedef_matrix_sqrt(inv_yy)

    if verbose:
    print("dot products...")
    arr = np.dot(invsqrt_xx, np.dot(sigma_xy, invsqrt_yy))

    if verbose:
    print("trying to take final svd")
    u, s, v = np.linalg.svd(arr)

    if verbose:
    print("computed everything!")

    return [u, np.abs(s), v], invsqrt_xx, invsqrt_yy, x_idxs, y_idxs


def sum_threshold(array, threshold):

    assert (threshold >= 0) and (threshold <= 1), "print incorrect threshold"

    for i in range(len(array)):
    if np.sum(array[:i])/np.sum(array) >= threshold:
      return i


def create_zero_dict(compute_dirns, dimension):
    return_dict = {}
    return_dict["mean"] = (np.asarray(0), np.asarray(0))
    return_dict["sum"] = (np.asarray(0), np.asarray(0))
    return_dict["cca_coef1"] = np.asarray(0)
    return_dict["cca_coef2"] = np.asarray(0)
    return_dict["idx1"] = 0
    return_dict["idx2"] = 0

    if compute_dirns:
    return_dict["cca_dirns1"] = np.zeros((1, dimension))
    return_dict["cca_dirns2"] = np.zeros((1, dimension))

    return return_dict


def get_cca_similarity(acts1, acts2, epsilon=0., threshold=0.98, compute_coefs=True, compute_dirns=False, verbose=True):
    # assert dimensionality equal
    assert acts1.shape[1] == acts2.shape[1], "dimensions don't match"
    # check that acts1, acts2 are transposition
    assert acts1.shape[0] < acts1.shape[1], ("input must be number of neurons"
                                           "by datapoints")
    return_dict = {}

    # compute covariance with numpy function for extra stability
    numx = acts1.shape[0]
    numy = acts2.shape[0]

    covariance = np.cov(acts1, acts2)
    sigmaxx = covariance[:numx, :numx]
    sigmaxy = covariance[:numx, numx:]
    sigmayx = covariance[numx:, :numx]
    sigmayy = covariance[numx:, numx:]

    # rescale covariance to make cca computation more stable
    xmax = np.max(np.abs(sigmaxx))
    ymax = np.max(np.abs(sigmayy))
    sigmaxx /= xmax
    sigmayy /= ymax
    sigmaxy /= np.sqrt(xmax * ymax)
    sigmayx /= np.sqrt(xmax * ymax)

    ([u, s, v], invsqrt_xx, invsqrt_yy,
    x_idxs, y_idxs) = compute_ccas(sigmaxx, sigmaxy, sigmayx, sigmayy,
                                  epsilon=epsilon,
                                  verbose=verbose)

    # if x_idxs or y_idxs is all false, return_dict has zero entries
    if (not np.any(x_idxs)) or (not np.any(y_idxs)):
    return create_zero_dict(compute_dirns, acts1.shape[1])

    if compute_coefs:

    # also compute full coefficients over all neurons
    x_mask = np.dot(x_idxs.reshape((-1, 1)), x_idxs.reshape((1, -1)))
    y_mask = np.dot(y_idxs.reshape((-1, 1)), y_idxs.reshape((1, -1)))

    return_dict["coef_x"] = u.T
    return_dict["invsqrt_xx"] = invsqrt_xx
    return_dict["full_coef_x"] = np.zeros((numx, numx))
    np.place(return_dict["full_coef_x"], x_mask,
             return_dict["coef_x"])
    return_dict["full_invsqrt_xx"] = np.zeros((numx, numx))
    np.place(return_dict["full_invsqrt_xx"], x_mask,
             return_dict["invsqrt_xx"])

    return_dict["coef_y"] = v
    return_dict["invsqrt_yy"] = invsqrt_yy
    return_dict["full_coef_y"] = np.zeros((numy, numy))
    np.place(return_dict["full_coef_y"], y_mask,
             return_dict["coef_y"])
    return_dict["full_invsqrt_yy"] = np.zeros((numy, numy))
    np.place(return_dict["full_invsqrt_yy"], y_mask,
             return_dict["invsqrt_yy"])

    # compute means
    neuron_means1 = np.mean(acts1, axis=1, keepdims=True)
    neuron_means2 = np.mean(acts2, axis=1, keepdims=True)
    return_dict["neuron_means1"] = neuron_means1
    return_dict["neuron_means2"] = neuron_means2

    if compute_dirns:
    # orthonormal directions that are CCA directions
    cca_dirns1 = np.dot(np.dot(return_dict["full_coef_x"],
                               return_dict["full_invsqrt_xx"]),
                        (acts1 - neuron_means1)) + neuron_means1
    cca_dirns2 = np.dot(np.dot(return_dict["full_coef_y"],
                               return_dict["full_invsqrt_yy"]),
                        (acts2 - neuron_means2)) + neuron_means2

    # get rid of trailing zeros in the cca coefficients
    idx1 = sum_threshold(s, threshold)
    idx2 = sum_threshold(s, threshold)

    return_dict["cca_coef1"] = s
    return_dict["cca_coef2"] = s
    return_dict["x_idxs"] = x_idxs
    return_dict["y_idxs"] = y_idxs
    # summary statistics
    return_dict["mean"] = (np.mean(s[:idx1]), np.mean(s[:idx2]))
    return_dict["sum"] = (np.sum(s), np.sum(s))

    if compute_dirns:
    return_dict["cca_dirns1"] = cca_dirns1
    return_dict["cca_dirns2"] = cca_dirns2

    return return_dict


def robust_cca_similarity(acts1, acts2, threshold=0.98, epsilon=1e-6, compute_dirns=True):

    for trial in range(num_cca_trials):
    try:
      return_dict = get_cca_similarity(acts1, acts2, threshold, compute_dirns)
    except np.LinAlgError:
      acts1 = acts1*1e-1 + np.random.normal(size=acts1.shape)*epsilon
      acts2 = acts2*1e-1 + np.random.normal(size=acts1.shape)*epsilon
      if trial + 1 == num_cca_trials:
        raise

    return return_dict

def neuron_act(train_dataloader, model):
    #Forward hook 
    
    h1 = model.bert.pooler.register_forward_hook(get_features('feats'))
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
    
    h1.remove() # close the hook 
    
    return FEATS 


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
    
    # choose the layers accordingly 
    feats1 = pd.read_csv("activations_pool_train.csv") # neuron activations from the pooler layer before dropout and classifier 
    feats2 = pd.read_csv("activations_enc11_train.csv") # neuron activations from the pooler layer before dropout and classifier 
    
    feats1.dropna(inplace=True)
    feats2.dropna(inplace=True)
    p1 = np.array(feats1)
    p2 = np.array(feats2)
    
    f_results = get_cca_similarity(p1[:, 1:].T, p2[:, 1:].T, epsilon=1e-11, verbose=False)

    # mean CCA score 
    print('CCA score: {:.4f}'.format(np.mean(f_results["cca_coef1"])))
    
    # plot for CCA 
    _plot_helper(f_results["cca_coef1"], "CCA coef idx", "CCA coef value")
    
    
    # SVCCA plot 
    # Mean subtract activations

    p1 = np.array(feats1)
    p2 = np.array(feats2)

    cacts1 = p1 - np.mean(p1, axis=1, keepdims=True)
    cacts2 = p2 - np.mean(p2, axis=1, keepdims=True)

    # Perform SVD
    U1, s1, V1 = np.linalg.svd(cacts1, full_matrices=False)
    U2, s2, V2 = np.linalg.svd(cacts2, full_matrices=False)

    svacts1 = np.dot(s1[:20]*np.eye(20), V1[:20])
    svacts2 = np.dot(s2[:20]*np.eye(20), V2[:20])

    svcca_results = get_cca_similarity(svacts1, svacts2, epsilon=1e-10, verbose=False)


    svcca_baseline = get_cca_similarity(svb1, svb2, epsilon=1e-10, verbose=False)
    print("NCBI: ", np.mean(svcca_results["cca_coef1"]))
    plt.plot(svcca_results["cca_coef1"], lw=2.0, label="NCBI")
    plt.xlabel("Sorted CCA Correlation Coeff Idx")
    plt.ylabel("CCA Correlation Coefficient Value")
    plt.legend(loc="best")
    plt.grid()
    
# %%time
if __name__ == '__main__':
    main()
    
    