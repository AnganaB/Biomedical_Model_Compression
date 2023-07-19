import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss
import numpy as np
import math


class ClfDistillLossFunction(nn.Module):
    def forward(self, num_labels, hidden, logits, bias, teacher_probs, labels):
        """
        :param hidden: [batch, n_features] hidden features from the model
        :param logits: [batch, n_classes] logit score for each class
        :param bias: [batch, n_classes] log-probabilties from the bias for each class
        :param labels: [batch] integer class labels
        :return: scalar loss
        """
        raise NotImplementedError()

class Plain(ClfDistillLossFunction):
    def forward(self, num_labels, hidden, logits, bias, teacher_probs, labels):
        return F.cross_entropy(logits.view(-1, num_labels), labels.view(-1))

class BiasProductBaseline(ClfDistillLossFunction):
    def forward(self, num_labels, hidden, logits, bias, teacher_probs, labels):
        lg_softmax_op = nn.LogSoftmax(dim=2)
        # print(logits.shape)
        # print(bias.shape)
        # applying log softmax to main models logits
        logits = logits.float() # in case we were in fp16 mode
        logits_log = lg_softmax_op(logits)
        # applying log softmax to bias models logits
        bias = bias.float() # to make sure dtype=float32 
        bias_log = torch.log(bias)

        return F.cross_entropy((logits_log + bias_log).view(-1, num_labels), labels.view(-1))

class LearnedMixinBaseline(ClfDistillLossFunction):
    def __init__(self, penalty):
        super().__init__()
        self.penalty = penalty
        self.bias_lin = torch.nn.Linear(768,1)

    def forward(self, num_labels, hidden, logits, bias, teacher_probs, labels):
        logits = logits.float()
        lg_softmax_op = nn.LogSoftmax(dim=2)
        logits_log = lg_softmax_op(logits)

        factor = self.bias_lin.forward(hidden)
        factor = factor.float()
        factor = F.softplus(factor)

        bias_log = torch.log(bias)
        bias_log = bias_log * factor

        bias_lp = F.softmax(bias_log, dim=2)
        entropy = -(torch.exp(bias_lp) * bias_lp).sum(2).mean(1)
        '''
        print(logits.shape)
        print(bias.shape)
        print(labels.shape)
        print(entropy.shape)
        '''
        loss = F.cross_entropy((logits_log + bias_log).view(-1, num_labels), labels.view(-1)) + self.penalty * entropy 

        loss = loss.mean()
        return loss


class ReweightBaseline(ClfDistillLossFunction):
    def forward(self, num_labels, hidden, logits, bias, teacher_probs, labels):
        logits = logits.float()
        loss = F.cross_entropy(logits.view(-1, num_labels), labels.view(-1), reduction='none')
        lab = labels.cpu()
        one_hot_labels = torch.eye(logits.size(2)).cuda()[lab]
        labels = labels.to('cuda', dtype = torch.float)
        weights = 1 - (one_hot_labels * bias).sum(2)
        return (weights.view(-1) * loss).sum() / weights.sum()

class DistillLoss(ClfDistillLossFunction):
    def forward(self, num_labels, hidden, logits, bias, teacher_probs, labels):
        softmax_op = torch.nn.Softmax(dim=2)
        probs = softmax_op(logits)

        example_loss = -(teacher_probs * probs.log()).sum(2)
        batch_loss = example_loss.mean()

        return batch_loss 

class SmoothedDistillLoss(ClfDistillLossFunction):
    def forward(self, num_labels,hidden, logits, bias, teacher_probs, labels):
        softmax_op = torch.nn.Softmax(dim=2)
        probs = softmax_op(logits) # probs from student model

        lab = labels.cpu()
        one_hot_labels = torch.eye(logits.size(2)).cuda()[lab]
        labels = labels.to('cuda', dtype = torch.float)
        weights = (1 - (one_hot_labels * bias).sum(2))
        weights = weights.unsqueeze(2).expand_as(teacher_probs)
        #print(one_hot_labels.shape)
        exp_teacher_probs = teacher_probs ** weights 
        norm_teacher_probs = exp_teacher_probs / exp_teacher_probs.sum(2).unsqueeze(2).expand_as(teacher_probs)

        example_loss = -(norm_teacher_probs * probs.log()).sum(2)
        batch_loss = example_loss.mean()

        return batch_loss
        
        
class NewDistillLoss(ClfDistillLossFunction):
    def forward(self, num_labels,hidden, logits, bias, teacher_probs, labels):
        softmax_op = torch.nn.Softmax(dim=2)
        probs = softmax_op(logits) # probs from student model

        lab = labels.cpu()
        #print(teacher_probs)
        one_hot_labels = torch.eye(logits.size(2)).cuda()[lab]
        labels = labels.to('cuda', dtype=torch.long)
        
        #one_hot_mask = one_hot_labels.int()
        #print('One hot')
        #print(one_hot_labels)
        #print(weights.shape)
        #weights = (one_hot_labels * (1 - bias))
        #weights = -1 * weights
        #print("neg")
        #print(weights)
        #one_hot_tensor = one_hot_labels.cpu()
        #one_hot_labels = one_hot_labels.to('cuda', dtype=torch.float)
        #zero_indices = np.where(one_hot_tensor == 0.0)
        #weights[zero_indices] = teacher_probs[zero_indices]
        
        #weights[~one_hot_mask] = teacher_probs[~one_hot_mask]
        #print('neg one')
        #print(teacher_probs[zero_indices])
        
        #print(one_hot_labels.shape)
        #print(weights)
        
        soft2teacher = softmax_op(teacher_probs)
        gt_vals = one_hot_labels * teacher_probs
        #teacher_greater06 = (gt_vals.max(dim=2).values >= 0.6)
        
        bias_gt = one_hot_labels * bias
        bias_mask_less085 = (bias_gt.max(dim=2).values <= 0.85)
        bias_mask_greater09 = (bias_gt.max(dim=2).values > 0.99)
        
        #final_mask =  teacher_greater06 * bias_mask_greater09

        norm_teacher_probs = soft2teacher
        
        norm_teacher_probs[bias_mask_less085] = one_hot_labels[bias_mask_less085]
        
        norm_teacher_probs[bias_mask_greater09] = teacher_probs[bias_mask_greater09]
        
        example_loss = -(norm_teacher_probs * probs.log()).sum(2)
        batch_loss = example_loss.mean()

        return batch_loss



