import cv2
import numpy as np
from torch.utils.data import Dataset

from chest_dcm_to_png import dicom_to_png

class Chest_Single_Data_Generator(Dataset):
    def __init__(self, img_size, input_img_paths, mask_paths, labels, transform=None):
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.mask_paths = mask_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.mask_paths)

    def get_labels(self):
        return self.labels

    def __getitem__(self, idx):
        input_img_path = self.input_img_paths[idx]
        mask_path = self.mask_paths[idx]

        temp_image = dicom_to_png(input_img_path, (self.img_size[0], self.img_size[1]))
        temp_mask =  cv2.imread(mask_path,0)
        temp_mask = cv2.resize(temp_mask, (self.img_size[0], self.img_size[1]))
        
        mask = img*(mask/255.)
        img = np.expand_dims(img, -1)
        mask = np.expand_dims(mask, -1)
        sample = {'image': img, 'mask': mask, 'label': self.labels[idx]}

        if self.transform:
            sample = self.transform(sample)
            sample['image'] /= 255.
            sample['mask'] /= 255.
        
        return sample['image'], sample['mask'], sample['label']