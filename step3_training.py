import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import torch.nn as nn
from MRA_dataset import *
from architecture.UNet3D import *
from architecture.AttentionUNET import AttentionUNet3D
from architecture.csnet_3d import CSNet3D
from architecture.RENet import RENet
from architecture.vnet import VNet
from architecture.Attention_DSV import *
from architecture.R2AttUNet3D import R2AttUNet
from losses_and_metrics import *
import utils
import pandas as pd
from architecture.R2AttUNet3D import *
if __name__ == '__main__':
    
    #torch.cuda.set_device(utils.get_avail_gpu()) # assign which gpu will be used (only linux works)
    use_visdom = True
    
    train_list = './csv_updated/train_list_e_brave.csv'
    val_list = './csv_updated/val_list_e_brave.csv'
    
    model_path = './models_trained'
    model_name = 'DSV_multi_Attention_with_enhancement_tversky_e_70_b_16' #remember to include the project title (e.g., ALV)
    checkpoint_name = 'latest_checkpoint_DSV_multi_attention_with_enhancement.tar'
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    num_classes = 2
    n_classes = 2
    classes = 2
    num_channels = 1
    n_channels = 1
    channels = 1
    num_epochs =70 
    num_workers = 8
    train_batch_size = 8
    val_batch_size = 1
    num_batches_to_print = 30
    soft= nn.Softmax(dim=1)
    if use_visdom:
        # set plotter
        global plotter
        plotter = utils.VisdomLinePlotter(env_name=model_name)
    
    # mkdir 'models'
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    
    # set dataset
    training_dataset = CT_Dataset(train_list)
    val_dataset = CT_Dataset(val_list)
    
    train_loader = DataLoader(dataset=training_dataset,
                              batch_size=train_batch_size,
                              shuffle=True,
                              num_workers=num_workers)
    
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=val_batch_size,
                            shuffle=False,
                            num_workers=num_workers)
    
    # set model
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3" 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  #  model = VNet().to(device)
   # model = UNet3D().to(device)
   # model  = RENet().to(device)
   # model = AttentionUNet3D(n_channels=1, n_classes=2).to(device, dtype=torch.float)
   #model = CSNet3D(classes=2, channels=1).to(device, dtype=torch.float)
   # model = R2AttUNet(in_channels=1, out_channels=2 )
    model = unet_CT_multi_att_dsv_3D(n_classes=2, in_channels=1).to(device)
  #  model = nn.DataParallel(model).cuda()
    opt = optim.Adam(model.parameters(), lr=0.0001, amsgrad=True)
    #scheduler = StepLR(opt, step_size=2, gamma=0.8)
    losses, mdsc = [], []
    val_losses, val_mdsc = [], []
    
    best_val_dsc = 0.0
    
    #cudnn
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    print('Training model...')
    class_weights = torch.Tensor([0.05, 20]).to(device, dtype=torch.float)
    for epoch in range(num_epochs):

        # training
        model.train()
        running_loss = 0.0
        running_dsc = 0.0
        loss_epoch = 0.0
        dsc_epoch = 0.0        
        for i_batch, batched_sample in enumerate(train_loader):

            # send mini-batch to device            
            inputs, labels = batched_sample['image'].to(device, dtype=torch.float), batched_sample['label'].to(device, dtype=torch.long)
            one_hot_labels = nn.functional.one_hot(labels[:, 0, :, :, :], num_classes=num_classes)
            #print("shape of label before permute ", one_hot_labels.shape)
            one_hot_labels = one_hot_labels.permute(0, 4, 1, 2, 3).to(device, dtype=torch.float)
          #  print("shape and type  of label after permute ", one_hot_labels.shape, one_hot_labels.type())
           # print("shape of inputs:", inputs.shape)
            # zero the parameter gradients
            opt.zero_grad()
         
            # forward + backward + optimize
            outputs = model(inputs)
            #print("shape of model:" , outputs.shape)
            #print("Type of prediction", outputs.type())
           # loss = FocalTverskyLoss(outputs, one_hot_labels)
            loss = TverskyLoss(outputs, one_hot_labels)
            #loss = Generalized_Dice_Loss(outputs, one_hot_labels,class_weights)
            dsc = weighting_DSC(outputs, one_hot_labels, class_weights)
            loss.backward()
            opt.step()
            
            # print statistics
            running_loss += loss.item()
            running_dsc += dsc.item()
            loss_epoch += loss.item()
            dsc_epoch += dsc.item()
            if i_batch % num_batches_to_print == num_batches_to_print-1:  # print every N mini-batches
                print('[--DSV-Attention--Epoch: {0}/{1}, Batch: {2}/{3}] loss: {4}, dsc: {5}'.format(epoch+1, num_epochs, i_batch+1, len(train_loader), running_loss/num_batches_to_print, running_dsc/num_batches_to_print))
                if use_visdom:
                    plotter.plot('loss', 'train', 'Loss', epoch+(i_batch+1)/len(train_loader), running_loss/num_batches_to_print)
                    plotter.plot('DSC', 'train', 'DSC', epoch+(i_batch+1)/len(train_loader), running_dsc/num_batches_to_print)
                running_loss = 0.0
                running_dsc = 0.0

        # record losses and dsc
        losses.append(loss_epoch/len(train_loader))
        mdsc.append(dsc_epoch/len(train_loader))
        
        #reset
        loss_epoch = 0.0
        dsc_epoch = 0.0
                
        # validation
        model.eval()
        with torch.no_grad():
            running_val_loss = 0.0
            running_val_dsc = 0.0
            val_loss_epoch = 0.0
            val_dsc_epoch = 0.0
            for i_batch, batched_val_sample in enumerate(val_loader):
                
                # send mini-batch to device
                inputs, labels = batched_val_sample['image'].to(device, dtype=torch.float), batched_val_sample['label'].to(device, dtype=torch.long)
                one_hot_labels = nn.functional.one_hot(labels[:, 0, :, :, :], num_classes=num_classes)
                one_hot_labels = one_hot_labels.permute(0, 4, 1, 2, 3).to(device, dtype=torch.float)
                
              
                outputs = model(inputs)
                #loss = FocalTverskyLoss(outputs, one_hot_labels)
                #loss = TverskyLoss(outputs, one_hot_labels)
                loss = Generalized_Dice_Loss(outputs, one_hot_labels,class_weights)
                dsc =  weighting_DSC(outputs, one_hot_labels, class_weights)
                
                running_val_loss += loss.item()
                running_val_dsc += dsc.item()
                val_loss_epoch += loss.item()
                val_dsc_epoch += dsc.item()
                
                if i_batch % num_batches_to_print == num_batches_to_print-1:  # print every N mini-batches
                    print('[Epoch: {0}/{1}, Val batch: {2}/{3}] val_loss: {4}, val_dsc: {5}'.format(epoch+1, num_epochs, i_batch+1, len(val_loader), running_val_loss/num_batches_to_print, running_val_dsc/num_batches_to_print))
                    running_val_loss = 0.0
                    running_val_dsc = 0.0
            
            # record losses and dsc
            val_losses.append(val_loss_epoch/len(val_loader))
            val_mdsc.append(val_dsc_epoch/len(val_loader))
            
            # reset
            val_loss_epoch = 0.0
            val_dsc_epoch = 0.0
            
            # output current status
            print('*****\nEpoch: {0}/{1}, loss: {2}, dsc: {3}\n         val_loss: {4}, val_dsc: {5}\n*****'.format(epoch+1, num_epochs, losses[-1], mdsc[-1], val_losses[-1], val_mdsc[-1]))
            if use_visdom:
                plotter.plot('loss', 'train', 'Loss', epoch+1, losses[-1])
                plotter.plot('DSC', 'train', 'DSC', epoch+1, mdsc[-1])
                plotter.plot('loss', 'val', 'Loss', epoch+1, val_losses[-1])
                plotter.plot('DSC', 'val', 'DSC', epoch+1, val_mdsc[-1])
            
        # save the 
        
        torch.save({'epoch': epoch+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'losses': losses,
                    'mdsc': mdsc,
                    'val_losses': val_losses,
                    'val_dsc': val_mdsc},
                    os.path.join(model_path, checkpoint_name))
        
        # save the best model
        if best_val_dsc < val_mdsc[-1]:
            best_val_dsc = val_mdsc[-1]
            torch.save({'epoch': epoch+1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': opt.state_dict(),
                        'losses': losses,
                        'mdsc': mdsc,
                        'val_losses': val_losses,
                        'val_dsc': val_mdsc},
                        os.path.join(model_path, '{}_best.tar'.format(model_name)))
            
        # save all losses and dsc data
        pd_dict = {'loss': losses, 'DSC': mdsc, 'val_loss': val_losses, 'val_DSC': val_mdsc}
        stat = pd.DataFrame(pd_dict)
        stat.to_csv('./csv_epoch_loss/losses_dsc_vs_epoch_DSV_multi_attention_with_enhancement.csv')
            
        # decay learning rate
#        scheduler.step()
