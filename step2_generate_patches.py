# importing the multiprocessing module 
import multiprocessing 
import os 
import pandas as pd
import numpy as np
import itk
from random import randint
import h5py
import torch
import glob

def get_stride(image_width, kernel_size):
    '''
    return proper stride that can slide all images with min number of steps (min number of patches)
           by the given image and kernel sizes
    '''
    n = image_width//kernel_size + 1
    stride = (image_width - kernel_size) // (n - 1)
    return stride

def generate_val_sample(image_list, h5_path, patch_size):
    # printing process id 
    print('PID: {0}, number of samples: {1}'.format(os.getpid(), len(image_list)))
    
    for idx in range(image_list.shape[0]):
        i_sample = image_list.iloc[idx]['ID']
        print('PID: {0} -- '.format(os.getpid()), image_list.iloc[idx]['image'][:])
        # read image
        itk_image = itk.imread(image_list.iloc[idx]['image'])
        itk_annotation = itk.imread(image_list.iloc[idx]['label'])
        np_image = itk.array_from_image(itk_image)
        np_annotation = itk.array_from_image(itk_annotation)
        np_annotation = np_annotation / np_annotation.max()
        # normalized
        np_image = (np_image - np_image.mean())/ np_image.std()
        
        # reshape
        np_image = np_image.reshape([1, np_image.shape[0], np_image.shape[1], np_image.shape[2]])
        np_annotation = np_annotation.reshape([1, np_annotation.shape[0], np_annotation.shape[1], np_annotation.shape[2]])
        tensor_image = torch.from_numpy(np_image)
        tensor_annotation = torch.from_numpy(np_annotation)
        
        # get patches with proper strides to slide all image
        image_patches = tensor_image.unfold(1, patch_size[0], get_stride(np_image.shape[1], patch_size[0])).unfold(2, patch_size[1], get_stride(np_image.shape[2], patch_size[1])).unfold(3, patch_size[2], get_stride(np_image.shape[3], patch_size[2]))
        annotation_patches = tensor_annotation.unfold(1, patch_size[0], get_stride(np_image.shape[1], patch_size[0])).unfold(2, patch_size[1], get_stride(np_image.shape[2], patch_size[1])).unfold(3, patch_size[2], get_stride(np_image.shape[3], patch_size[2]))
        image_patches = image_patches.reshape(-1, 1, patch_size[0], patch_size[1], patch_size[2])
        annotation_patches = annotation_patches.reshape(-1, 1, patch_size[0], patch_size[1], patch_size[2])
        patch_image = image_patches.numpy()
        patch_label = annotation_patches.numpy()
        
        # save to h5
        if not os.path.exists(h5_path):
            os.makedirs(h5_path)
            
        for i_patch in range(image_patches.shape[0]):
            patch_file_name = os.path.join(h5_path, 'val_sample_{0}_patch_{1}x{2}x{3}_{4}_labelled.h5'.format(i_sample, patch_size[0], patch_size[1], patch_size[2], i_patch))
            
            #check old patch file
            if os.path.isfile(patch_file_name):
              os.remove(patch_file_name)
              
             #output h5 file
            with h5py.File(patch_file_name, 'w') as f:
              f['image'] = patch_image[i_patch, :, :, :, :]
              f['label'] = patch_label[i_patch, :, :, :, :]
        
  
def generate_random_patches(image_list, h5_path, patch_size, target_num_patches_each_label, valid_pct): 
    # printing process id 
    print('PID: {0}, number of samples: {1}'.format(os.getpid(), len(image_list)))
    
    for idx in range(image_list.shape[0]):
        i_sample = image_list.iloc[idx]['ID']
        print('PID: {0} -- '.format(os.getpid()), image_list.iloc[idx]['image'][:])
        # read image
        itk_image = itk.imread(image_list.iloc[idx]['image'])
        itk_annotation = itk.imread(image_list.iloc[idx]['label'])
        np_image = itk.array_from_image(itk_image)
        np_annotation = itk.array_from_image(itk_annotation)
        
        #normalized
        np_image = (np_image - np_image.mean())/ np_image.std()
        threshold_gray_value = np_image.mean() - 2*np_image.std()
        
        # get valid range
        valid_range = np.zeros([3, 2], dtype=np.int32)
        valid_range[0, 1] = np_image.shape[0]-patch_size[0]
        valid_range[1][1] = np_image.shape[1]-patch_size[1]
        valid_range[2][1] = np_image.shape[2]-patch_size[2]
        
        patch_image = np.zeros([target_num_patches_each_label.sum(), 1, patch_size[0], patch_size[1], patch_size[2]]) #Batch x C x W x D x H 
        patch_label = np.zeros([target_num_patches_each_label.sum(), 1, patch_size[0], patch_size[1], patch_size[2]]) #Batch x C x W x D x H 
        
        patch_volume = patch_size[0]*patch_size[1]*patch_size[2]
        
        # randomly sampled
        i_num_valid_patches = 0
        visited_location = []
        for i_label in range(target_num_patches_each_label.shape[0]):
            
            i_num_valid_patches_each_label = 0
            while i_num_valid_patches_each_label < target_num_patches_each_label[i_label]:
                k = randint(valid_range[0, 0], valid_range[0, 1])
                j = randint(valid_range[1, 0], valid_range[1, 1])
                i = randint(valid_range[2, 0], valid_range[2, 1])
                i_location = [k, j, i] # bottom left corner, i.e., 000
                
                if not i_location in visited_location:
                    i_patch_image =  np_image[k:(k+patch_size[0]),
                                            j:(j+patch_size[1]),
                                            i:(i+patch_size[2])]
                    i_patch_label =  np_annotation[k:(k+patch_size[0]),
                                                 j:(j+patch_size[1]),
                                                 i:(i+patch_size[2])]
                    
                    if (np.sum(i_patch_label==i_label) > patch_volume*valid_pct) and (np.sum(i_patch_image > threshold_gray_value)):
                        visited_location.append(i_location) # valid visit
                        
                        patch_image[i_num_valid_patches, 0, :, :, :] = i_patch_image
                        patch_label[i_num_valid_patches, 0, :, :, :] = i_patch_label
                        
                        i_num_valid_patches_each_label += 1
                        i_num_valid_patches += 1
#                        print('Current total number of patches: {0}\n  For label of {1}: patch No. {2}'.format(i_num_valid_patches, i_label, i_num_valid_patches_each_label))
            
        # shuffle
        randnum = list(range(patch_image.shape[0]))
        np.random.shuffle(randnum)
        patch_image = patch_image[randnum, :]
        patch_label = patch_label[randnum, :]
        
        # save to h5
        if not os.path.exists(h5_path):
            os.makedirs(h5_path)
            
        for i_patch in range(patch_image.shape[0]):
            patch_file_name = os.path.join(h5_path, 'sample_{0}_patch_{1}x{2}x{3}_{4}.h5'.format(i_sample, patch_size[0], patch_size[1], patch_size[2], i_patch))
            
            #check old patch file
            if os.path.isfile(patch_file_name):
              os.remove(patch_file_name)
              
             #output h5 file
            with h5py.File(patch_file_name, 'w') as f:
              f['image'] = patch_image[i_patch, :, :, :, :]
              f['label'] = patch_label[i_patch, :, :, :, :]

def generate_val_sample_t(image_list, h5_path, patch_size):
    # printing process id
    print('PID: {0}, number of samples: {1}'.format(os.getpid(), len(image_list)))

    for idx in range(image_list.shape[0]):
        i_sample = image_list.iloc[idx]['ID']
        print('PID: {0} -- '.format(os.getpid()), image_list.iloc[idx]['image'][:])
        # read image
        itk_image = itk.imread(image_list.iloc[idx]['image'])
        itk_annotation = itk.imread(image_list.iloc[idx]['label'])
        np_image = itk.array_from_image(itk_image)
        np_annotation = itk.array_from_image(itk_annotation)
        np_annotation = np_annotation / np_annotation.max()
        # normalized
        np_image = (np_image - np_image.mean()) / np_image.std()

        # reshape
        np_image = np_image.reshape([1, np_image.shape[0], np_image.shape[1], np_image.shape[2]])
        np_annotation = np_annotation.reshape(
            [1, np_annotation.shape[0], np_annotation.shape[1], np_annotation.shape[2]])
        tensor_image = torch.from_numpy(np_image)
        tensor_annotation = torch.from_numpy(np_annotation)

        # get patches with proper strides to slide all image
        image_patches = tensor_image.unfold(1, patch_size[0], get_stride(np_image.shape[1], patch_size[0])).unfold(2,patch_size[1],          get_stride(np_image.shape[2],patch_size[1])).unfold(3, patch_size[2], get_stride(np_image.shape[3], patch_size[2]))
        annotation_patches = tensor_annotation.unfold(1, patch_size[0],get_stride(np_image.shape[1], patch_size[0])).unfold(2,patch_size[1],get_stride(np_image.shape[2],patch_size[1])).unfold(3, patch_size[2], get_stride(np_image.shape[3], patch_size[2]))
        
        image_patches = image_patches.reshape(-1, 1, patch_size[0], patch_size[1], patch_size[2])
        annotation_patches = annotation_patches.reshape(-1, 1, patch_size[0], patch_size[1], patch_size[2])
        patch_image = image_patches.numpy()
        patch_label = annotation_patches.numpy()

        # save to h5
        if not os.path.exists(h5_path):
            os.makedirs(h5_path)

        for i_patch in range(image_patches.shape[0]):
            patch_file_name = os.path.join(h5_path,
                                           'sample_{0}_patch_{1}x{2}x{3}_{4}_labelled.h5'.format(i_sample, patch_size[0],
                                                                                            patch_size[1],
                                                                                            patch_size[2], i_patch))

            # check old patch file
            if os.path.isfile(patch_file_name):
                os.remove(patch_file_name)

            # output h5 file
            with h5py.File(patch_file_name, 'w') as f:
                f['image'] = patch_image[i_patch, :, :, :, :]
                f['label'] = patch_label[i_patch, :, :, :, :]

  
if __name__ == "__main__": 
    
    #inputs
    num_workers = 8
    csv_file = '/home/bravo/workspace/Farhan/UNet3D-master/data_list_labelled.csv'
    csv_path = '/home/bravo/workspace/Farhan/UNet3D-master/data_list_labelled.csv'
    h5_path = '/home/bravo/DATA1/Farhan/MRA_Data/patches/patches_labelled'
    h5_list_path = './' # train_list.txt and val_list.txt
    
    #remove old patches (h5 files) in h5_path
    old_file_list = glob.glob(os.path.join(h5_path, "*.h5"))
    for f in old_file_list:
        os.remove(os.path.join(h5_path, f))
    
    train_patch_size = np.array([64, 64, 64])
    val_patch_size = np.array([128, 128, 128])
    target_num_patches_each_label = np.array([2, 3, 3])
    valid_pct = 0.01
    train_size = 0.9
    
    csv_file = os.path.join(csv_path, csv_file)
    image_list = pd.read_csv(csv_file)
    image_list['ID'] = image_list.index + 1 # add 'ID' column
    
    sample_idx = image_list['ID'].tolist()
    np.random.shuffle(sample_idx) #shuffle sample_list
    split_idx = int(np.round(train_size*len(sample_idx)))
    train_idx, val_idx = np.split(sample_idx, [split_idx])
    
    train_image_list_split = np.array_split(image_list.iloc[train_idx-1], num_workers)
    val_image_list_split = np.array_split(image_list.iloc[val_idx-1], num_workers)
  
    # generate train_patch h5
    p_list = []
    for i_worker in range(num_workers):
        
        p = multiprocessing.Process(target=generate_val_sample_t, args=(train_image_list_split[i_worker],
                                                                   h5_path,
                                                                   train_patch_size))
        p.start()
        p_list.append(p)
    
    # wait until all processors done    
    for p in p_list:
        p.join()
        
    # generate val_sample h5
    p_list = []
    for i_worker in range(num_workers):
        
        p = multiprocessing.Process(target=generate_val_sample, args=(val_image_list_split[i_worker],
                                                                      h5_path,
                                                                      val_patch_size, )) 
        p.start()
        p_list.append(p)
        
    # wait until all processors done
    for p in p_list:
        p.join()
        
    
    # training list
    train_list = glob.glob('/home/bravo/DATA1/Farhan/MRA_Data/patches/sample_*.h5')
    train_pd = pd.DataFrame(train_list)
    train_pd.to_csv('train_list.csv', header=False, index=False)
            
    # validation list
    val_list = glob.glob('/home/bravo/DATA1/Farhan/MRA_Data/patches/val_*.h5')
    val_pd = pd.DataFrame(val_list)
    val_pd.to_csv('val_list.csv', header=False, index=False)
