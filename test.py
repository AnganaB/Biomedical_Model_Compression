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
from transformers import AutoTokenizer, AutoConfig, AutoModelForTokenClassification, AutoModel, BertPreTrainedModel
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

output_file = 'output_file_b1_ncbi'
input_path = '/raid/data/kay/mounted/IITG/'
output_path = 'resources'
log_soft = F.log_softmax
tokenizer_dir = "/raid/data/kay/mounted/IITG/tokenizer"
model_dir = "/raid/data/kay/mounted/IITG/model"
config_dir = "/raid/data/kay/mounted/IITG/config"
# output_model_directory = "/raid/data/kay/mounted/IITG/"
model_directory = "/raid/data/kay/mounted/IITG/"
tokenizer_directory = "/raid/data/kay/mounted/IITG/"
print(torch.version.cuda)
MAX_LEN = 310# suitable for all datasets
MAX_GRAD_NORM = 10
BATCH_SIZE = 16
LEARNING_RATE = 3e-5
MAX_LEN = 310
BATCH_SIZE = 16
count_labels = []

def inference(model, dataloader, tokenizer, device, id2label):
    model.eval()
    pred_lst = []
    test_loss = 0
    nb_test_steps = 0
    test_accuracy = 0
    for idx, batch in enumerate(dataloader):
        ids = batch['ids'].to(device, dtype=torch.long)
        mask = batch['mask'].to(device, dtype=torch.long)
        targets = batch['targets'].to(device, dtype=torch.long)

        outputs = model(input_ids=ids, attention_mask=mask, labels=targets)
        # outputs = model(model.num_labels, ids, mask, labels=targets)

        loss, inference_logits = outputs[0], outputs[1]
        test_loss += loss.item()
        nb_test_steps += 1
        if_logits = inference_logits
        #if_logits = F.softmax(inference_logits, dim=2)
        flattened_targets = targets.view(-1)
        flattened_predictions = if_logits.view(-1)
        #flattened_predictions = torch.argmax(active_logits, axis=1)
        active_accuracy = mask.view(-1) == 1
        targets = torch.masked_select(flattened_targets, active_accuracy)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)

        tmp_test_accuracy = accuracy_score(targets.cpu().numpy(), predictions.cpu().numpy())
        test_accuracy += tmp_test_accuracy

        #inference_logits = F.softmax(inference_logits, dim=2)
        #inference_ids = torch.argmax(inference_logits, dim=2)
        for i in range(ids.shape[0]):
            tmp_labels = []
            tmp_test_tokens = tokenizer.convert_ids_to_tokens(ids[i])

            tmp_label_ids = inference_logits[i]
            for index, tok in enumerate(tmp_test_tokens):
                if tok in ['[CLS]', '[SEP]', '[PAD]']:
                    continue 
                else:
                    tmp_labels.append(id2label[tmp_label_ids[index].item()])
            
            pred_lst.append(tmp_labels)

    test_accuracy = test_accuracy / nb_test_steps
    matrix = confusion_matrix(targets.cpu().numpy(), predictions.cpu().numpy())
    print(matrix)
    print(matrix.diagonal()/matrix.sum(axis=1))
    print(f'\t Test accuracy: {test_accuracy}')
    print("F1 score:", f1_score(targets.cpu().numpy(), predictions.cpu().numpy(), average="macro"))
    print("Precision Score:", precision_score(targets.cpu().numpy(), predictions.cpu().numpy(), average="macro"))
    print("Recall Score:" ,recall_score(targets.cpu().numpy(), predictions.cpu().numpy(), average="macro"))
    return pred_lst


def generate_prediction_file(pred_labels, tokens, dataset_name, tokenizer, output_file):
    #p_name = 'preds_' + model_name + '.txt'
    #output_file = os.path.join(dataset_name, p_name)
    labels = pred_labels[0]
    i = 1
    j = 0
    with open(output_file, 'w') as fh:
        for tok in tokens:
            if isinstance(tok, float):
                fh.write('\n')
                if i >= len(pred_labels):
                    break
                labels = pred_labels[i]
                i += 1
                j = 0
            elif j < len(labels):
                sub_words = tokenizer.tokenize(tok)
                fh.write(f'{tok}\t{labels[j]}\n')
                j += len(sub_words)
            else:
                fh.write(f'{tok}\tO\n')

def main():

    input_model_path = os.path.join(input_path, model_directory)
    input_tokenizer_path = os.path.join(input_path, tokenizer_directory)
    loss_fn = "Plain"
    # read data
    train_data, test_data = read_data(dataset_name)
    # get a dict for label and its id
    id2label,label2id = IdToLabelAndLabeltoId(train_data)
    #get list of sentence and associated label
    test_sent, test_label = convert_to_sentence(test_data)
    
    device = 'cuda' if cuda.is_available() else 'cpu'
    # loading tokenizer
    loss_fn = clf_distill_loss_functions.Plain()
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_directory
    )
    model = MyModel.from_pretrained(
        model_directory,loss_fn = loss_fn
    )

    num_labels = len(id2label)
    test_df = {'Sentence':test_sent, 'Labels':test_label}
    test_df = pd.DataFrame(test_df)

    test_params = {'batch_size': BATCH_SIZE,
                    'shuffle': False,
                    }

    test_dataset = dataset(test_df, tokenizer, MAX_LEN, label2id, id2label)
    test_dataloader = DataLoader(test_dataset, **test_params)

  
    model.to(device)

    # getting predictions
    print("Testing started")
    pred_labels = inference(model, test_dataloader, tokenizer, device, id2label)
    # writing to file
    generate_prediction_file(pred_labels, test_data['Tokens'].tolist(), dataset_name, tokenizer, output_file)

# %%time
if __name__ == '__main__':
    main()
