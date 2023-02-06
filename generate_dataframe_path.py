import pandas as pd

if __name__ == '__main__':
    
    image_path = '/home/bravo/DATA1/Farhan/MRA_Data/MR_68/Normal_Data/'
    image_path_labels = '/home/bravo/DATA1/Farhan/MRA_Data/MR_68/labels/'
    
    image_filename = []
    annotation_filename = []
    ##Number of Imaages
    for i in range(1, 86):
        image_filename.append('Normal0{0}-MRA.nii'.format(i))
        annotation_filename.append('Normal0{0}-MRA.nii'.format(i))
        
    df = pd.DataFrame(list(zip(image_filename, annotation_filename)), columns=['image', 'label'])
    df['image'] = image_path+ df['image']
    df['label'] = image_path_labels+ df['label']
    df.to_csv('./data_list_labelled.csv', header=True, index=False)
        
        
        
