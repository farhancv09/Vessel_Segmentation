import torch
import numpy as np
import torch.nn as nn

from torch.nn import functional as F
def weighting_DSC(y_pred, y_true, class_weights, smooth = 1.0):
    '''
    inputs:
        y_pred [batch, n_classes, x, y, z] probability
        y_true [batch, n_classes, x, y, z] one-hot code
        class_weights
        smooth = 1.0
    '''
    smooth = 1.
    mdsc = 0.0
    n_classes = y_pred.shape[1] # for PyTorch data format
    
    # convert probability to one-hot code    
    max_idx = torch.argmax(y_pred, dim=1, keepdim=True)
    one_hot = torch.zeros_like(y_pred)
    one_hot.scatter_(1, max_idx, 1)

    for c in range(0, n_classes):
        pred_flat = one_hot[:, c].reshape(-1)
        true_flat = y_true[:, c].reshape(-1)
        intersection = (pred_flat * true_flat).sum()
        w = class_weights[c]/class_weights.sum()
        mdsc += w*((2. * intersection + smooth) / (pred_flat.sum() + true_flat.sum() + smooth))
        
    return mdsc

def Generalized_Dice_Loss(y_pred, y_true, class_weights, smooth = 1.0):
    '''
    inputs:
        y_pred [batch, n_classes, x, y, z] probability
        y_true [batch, n_classes, x, y, z] one-hot code
        class_weights
        smooth = 1.0
    '''
    smooth = 1.
    loss = 0.
    n_classes = y_pred.shape[1]
    for c in range(0, n_classes): #pass 0 because 0 is background
        pred_flat = y_pred[:, c].reshape(-1)
        true_flat = y_true[:, c].reshape(-1)
        tp = (pred_flat *true_flat).sum()
        fp = ((1-true_flat)*pred_flat).sum()
        fn = (true_flat * (1 - pred_flat)).sum()
        #print(pred_flat.type())
        #print(true_flat.type())
       # pred_flat = torch.FloatTensor(pred_flat.cpu()) 
       # true_flat =true_flat.type(torch.FloatTensor)
        #true_flat=true_flat.cuda()
        #true_flat =true_flat.type(torch.FloatTensor)
        #print(pred_flat.type())
       # print(true_flat.type())
        intersection = (pred_flat * true_flat).sum()
       
        # with weight
        w = class_weights[c]/class_weights.sum()
        loss += w*(1 - ((2. * intersection + smooth) /
                         (pred_flat.sum() + true_flat.sum() + smooth)))
       
    return loss

def TverskyLoss(y_pred, y_true,alpha = 0.3, beta = 0.7, smooth=1.0):
    smooth = 1.
    loss = 0.
    n_classes = y_pred.shape[1]
    for c in range(0, n_classes):
        
           #pass 0 because 0 is background
        pred_flat = y_pred[:, c].reshape(-1)
        true_flat = y_true[:, c].reshape(-1)
        tp = (pred_flat *true_flat).sum()
        fp = ((1-true_flat)*pred_flat).sum()
        fn = (true_flat * (1 - pred_flat)).sum()
        loss += (1- ((tp+smooth)/(tp + alpha * fp +beta * fn +smooth))) 
    return loss

    
def FocalTverskyLoss(y_pred, y_true,alpha = 0.3, beta = 0.7, smooth=1.0, gamma = 0.75):
    smooth = 1.
    loss = 0.
    n_classes = y_pred.shape[1]
    for c in range(0, n_classes):
            #pass 0 because 0 is background
        pred_flat = y_pred[:, c].reshape(-1)
        true_flat = y_true[:, c].reshape(-1)
        tp = (pred_flat *true_flat).sum()
        fp = ((1-true_flat)*pred_flat).sum()
        fn = (true_flat * (1 - pred_flat)).sum()
        tversky_loss = (1- ((tp+smooth)/(tp + alpha * fp +beta * fn +smooth)))
        loss += tversky_loss**gamma
        
    return loss
          
    

def DSC(y_pred, y_true, ignore_background=True, smooth = 1.0):
    '''
    inputs:
        y_pred [n_classes, x, y, z] one-hot code
        y_true [n_classes, x, y, z] one-hot code
    '''
    smooth = 1.
    n_classes = y_pred.shape[0]
    dsc = []
    if ignore_background:
        for c in range(1, n_classes): #pass 0 because 0 is background
            pred_flat = y_pred[c, :].reshape(-1)
            true_flat = y_true[c, :].reshape(-1)
            true_flat =true_flat.type(torch.FloatTensor)
            intersection = (pred_flat * true_flat).sum()
            dsc.append(((2. * intersection + smooth) / (pred_flat.sum() + true_flat.sum() + smooth)))
            
        dsc = np.asarray(dsc)
    else:
        for c in range(0, n_classes):
            pred_flat = y_pred[c, :].reshape(-1)
            true_flat = y_true[c, :].reshape(-1)
            intersection = (pred_flat * true_flat).sum()
            dsc.append(((2. * intersection + smooth) / (pred_flat.sum() + true_flat.sum() + smooth)))
            
        dsc = np.asarray(dsc)
        
    return dsc

class FocalTverskyLoss2(nn.Module):
    ALPHA = 0.3
    BETA = 0.7
    GAMMA = 0.75

    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.5, beta=0.5, gamma=0.5):
        # comment out if your model contains a sigmoid or equivalent activation layer
        targets_layered = torch.zeros_like(inputs)
        for idx in range(inputs.shape[1]):
            targets_layered[:, idx, :, :][targets == idx] = 1
        inputs = F.sigmoid(inputs)
        targets = targets_layered
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        tp = (inputs * targets).sum()
        fp = ((1 - targets) * inputs).sum()
        fn = (targets * (1 - inputs)).sum()

        tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
        focal_tversky = (1 - tversky) ** gamma

        return focal_tversky