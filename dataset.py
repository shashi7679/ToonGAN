import os
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import config
from utils import check_cords, face_parsing
import torchvision.transforms as transforms
import skimage

############## Augmentations ###############
both_transform = A.Compose(
    [A.Resize(width=config.IMAGE_SIZE, height=config.IMAGE_SIZE),], additional_targets={"image0": "image"},
)

transform_only_input = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

transform_only_mask = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

transform_test_data = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


class Face_Toon_Dataset(Dataset):
    def __init__(self,root):
        self.root = root
        self.original_root = root + 'original/'
        self.toon_root = root + 'toon/'
        list_files = os.listdir(self.original_root)
        #### Removing '.ipynb_checkpoints' from the list
        # list_files.remove('.ipynb_checkpoints')
        self.n_samples = list_files
        
            
    
    def __len__(self):
        return len(self.n_samples)
    
    def __getitem__(self,idx):
        try:
            if torch.is_tensor(idx):
                idx = idx.tolist()
            image_name = self.n_samples[idx]
            # print(image_name)

            original_image_path = os.path.join(self.original_root,image_name)
            toon_image_path = os.path.join(self.toon_root,image_name)

            original_image = np.asarray(Image.open(original_image_path).convert('RGB'))
            toon_image = np.asarray(Image.open(toon_image_path).convert('RGB'))

            # augmentations = both_transform(image=original_image, image0=toon_image)
            # input_image = augmentations["image"]
            # target_image = augmentations["image0"]

            # original_image = transform_only_input(image=input_image)["image"]
            # toon_image = transform_only_mask(image=target_image)["image"]

            original_image = transform_test_data(original_image)
            toon_image = transform_test_data(toon_image)

            return (original_image, toon_image)
        except:
            if torch.is_tensor(idx):
                idx = idx.tolist()
            image_name = self.n_samples[idx]
            #print(self.n_samples)
            image_path = os.path.join(self.original_root,image_name)
            print(image_path)
            pass


class Test_Faces(Dataset):
    def __init__(self,root,transform=None):
        self.root = root
        list_files = os.listdir(self.root)
        self.n_samples = list_files
        self.transform = transform
         
    def __len__(self):
        return len(self.n_samples)
    
    def __getitem__(self,idx):
            if torch.is_tensor(idx):
                idx = idx.tolist()
            image_name = self.n_samples[idx]

            image_path = os.path.join(self.root,image_name)

            # print(image_path)

            original_image = np.asarray(Image.open(image_path).convert('RGB'))
            y_shape, x_shape = original_image.shape[0], original_image.shape[1]
            ################### Extracting Face #######################
            detection = config.DETECTOR.detect(original_image)
            # print(detection.shape)
            xmin, ymin, xmax, ymax, c = detection[0]
            height = ymax-ymin
            width = xmax-xmin
            xmin_mod_1, ymin_mod_1, xmax_mod_1, ymax_mod_1 = int(xmin - width/6), int(ymin - height/4), int(xmax + width/6), int(ymax + width/15)

                    #------ Checking Cordinates to keep it within valid ranges [0,480] ------#
            xmin_mod_1, ymin_mod_1, xmax_mod_1, ymax_mod_1 = check_cords(xmin_mod_1, ymin_mod_1, xmax_mod_1, ymax_mod_1, x_shape, y_shape)

            face_image = original_image[ymin_mod_1:ymax_mod_1, xmin_mod_1:xmax_mod_1]
            # print(xmin_mod_1, ymin_mod_1, xmax_mod_1, ymax_mod_1)
            # print(face_image.shape)

            original_face_image = face_image   

            face_image = face_parsing(face_image)

            if self.transform:
                face_image = self.transform(face_image)
            else:
                face_image = transform_test_data(face_image)

            return (face_image, (x_shape, y_shape), (xmin_mod_1, ymin_mod_1, xmax_mod_1, ymax_mod_1), original_image, original_face_image)
    
    
            
if __name__=="__main__":
    # dataset = Face_Toon_Dataset("./pix2pix_data/train/")
    # loader = DataLoader(dataset, batch_size=1)
    # for x,y in loader:
    #     print("X Shape :-",x.shape)
    #     print("Y Shape :-",y.shape)
    #     # save_image(x*0.5 + 0.5,"original.png")
    #     # save_image(y*0.5 + 0.5,"toon.png") 
    test_dataset = Test_Faces('./toon_pix_data/test/')
    sample,_,_,_,original = test_dataset[1]
    sample = sample * 0.5 + 0.5
    save_image(sample,"temp_parsed.png")
    skimage.io.imsave('temp_parsed_1.png', original)
    print(sample.shape)
                  