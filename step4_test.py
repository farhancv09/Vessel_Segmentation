import os
import numpy as np
import torch
import torch.nn as nn
from UNet3D import *
import utils
import itk
import pandas as pd
from losses_and_metrics import *

def get_stride(image_width, kernel_size):
    '''
    return proper stride that can slide all images with min number of steps (min number of patches)
           by the given image and kernel sizes
    '''
    n = image_width//kernel_size + 1
    stride = (image_width - kernel_size) // (n - 1)
    return stride

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3" 
    torch.cuda.set_device(utils.get_avail_gpu()) # assign which gpu will be used (only linux works)
      
    model_path = './models'
    #model_name = 'checkpoint.tar'
    model_name = 'ALV_unet3d_patch64x64x64_3000_3labels_30samples_best.tar'
    
    image_path = '/data/data/Beijing_CBCT_Unilateral_Cleft_Lip_and_Palate/GroundTruth/flip_Res0p4_smoothed'
    test_list = [2, 6, 8, 9, 12, 18]
    test_image_filename = 'NORMAL0{0}_cbq-n3-7.hdr'
    test_label_filename = 'NORMAL0{0}-ls-corrected-ordered-smoothed.nrrd'
    test_path = './test'
    if not os.path.exists(test_path):
        os.mkdir(test_path)
    
    num_classes = 2
    num_channels = 1
    patch_size = np.array([128, 128, 128])
          
    # set model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet3D(in_channels=num_channels, out_channels=num_classes).to(device, dtype=torch.float)
    
    # load trained model
    checkpoint = torch.load(os.path.join(model_path, model_name), map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    del checkpoint
    model = model.to(device, dtype=torch.float)
    
    
    #cudnn
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    model = nn.DataParallel(model).cuda()
    # Testing
    dsc_label_1 = []
    dsc_label_2 = []
    
    print('Testing')
    model.eval()

    with torch.no_grad():
        for i_sample in test_list:
            
            print('Predicting Sample filename: {}'.format(test_image_filename.format(i_sample)))
            # read image and label (annotation)
            itk_image = itk.imread(os.path.join(image_path, test_image_filename.format(i_sample)))
            itk_label = itk.imread(os.path.join(image_path, test_label_filename.format(i_sample)))
            np_image = itk.array_from_image(itk_image)
            np_label = itk.array_from_image(itk_label)
            predicted_label = np.zeros(np_label.shape)
        
            # normalized
            np_image = (np_image - np_image.mean())/ np_image.std()
            np_image = np_image.reshape([1, np_image.shape[0], np_image.shape[1], np_image.shape[2]])
            
            # numpy -> torch.tensor
            tensor_image = torch.from_numpy(np_image).to(device, dtype=torch.float)
            
            # get stirde, will use when fold
            stride_1 = get_stride(np_image.shape[1], patch_size[0])
            stride_2 = get_stride(np_image.shape[2], patch_size[1])
            stride_3 = get_stride(np_image.shape[3], patch_size[2])
            
            # get num of strides
            n1 = (np_image.shape[1]-patch_size[0])/stride_1 + 1
            n2 = (np_image.shape[2]-patch_size[1])/stride_2 + 1
            n3 = (np_image.shape[3]-patch_size[2])/stride_3 + 1
            
            # create patches covering the entire image
            image_patches = tensor_image.unfold(1, patch_size[0], get_stride(np_image.shape[1], patch_size[0])).unfold(2, patch_size[1], get_stride(np_image.shape[2], patch_size[1])).unfold(3, patch_size[2], get_stride(np_image.shape[3], patch_size[2]))
            image_patches = image_patches.reshape(-1, 1, patch_size[0], patch_size[1], patch_size[2])
           
            patch_output = np.zeros(image_patches.shape)
            for i_patch in range(image_patches.shape[0]):
                tensor_prob_output = model(image_patches[i_patch, :, :, :, :,].view(-1, num_channels, patch_size[0], patch_size[1], patch_size[2]).to(device, dtype=torch.float)).detach()
                patch_prob_output = tensor_prob_output.cpu().numpy()
                
                for i_label in range(num_classes):
                    patch_output[i_patch, np.argmax(patch_prob_output[:], axis=1)==i_label] = i_label
                    
            # squeeze
            patch_output = np.squeeze(patch_output, axis=1)
            
            # self fold function: combine patch results
            i_patch = 0
            for k in range(int(n1)):
                for j in range(int(n2)):
                    for i in range(int(n3)):
                        predicted_label[0+int(stride_1*k):patch_size[0]+int(stride_1*k), 0+int(stride_2*j):patch_size[1]+int(stride_2*j), 0+int(stride_3*i):patch_size[2]+int(stride_3*i)] = patch_output[i_patch, :]
                        i_patch += 1
                        
            # output test result
            itk_predict_label = itk.image_view_from_array(predicted_label)
            itk.imwrite(itk_predict_label, os.path.join(test_path, 'Sample_{}_predicted.nrrd'.format(i_sample)))
            
            # convert predict result and label to one-hot maps
            tensor_predicted_label = torch.from_numpy(predicted_label)
            tensor_np_label = torch.from_numpy(np_label)
            tensor_predicted_label = tensor_predicted_label.long()
            tensor_np_label = tensor_np_label.long()
            one_hot_predicted_label = nn.functional.one_hot(tensor_predicted_label[:, :, :], num_classes=num_classes)
            one_hot_predicted_label = one_hot_predicted_label.permute(3, 0, 1, 2)
            one_hot_labels = nn.functional.one_hot(tensor_np_label[:, :, :], num_classes=num_classes)
            one_hot_labels = one_hot_labels.permute(3, 0, 1, 2)
            
            # calculate DSC
            dsc = DSC(one_hot_predicted_label, one_hot_labels)
            dsc_label_1.append(dsc[0])
            dsc_label_2.append(dsc[1])
            print('\tLabel 1: {}; Label 2: {}'.format(dsc[0], dsc[1]))
        
    # output all DSCs
    all_dsc = pd.DataFrame(list(zip(test_list, dsc_label_1, dsc_label_2)), columns=['Sample', 'Label 1', 'Label 2'])
    print(all_dsc)
    print(all_dsc.describe())
    all_dsc.to_csv(os.path.join(test_path, 'test_DSC_report.csv'), header=True, index=False)
            
    
            
