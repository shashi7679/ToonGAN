import os
import torch
from PIL import Image
import config
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils import check_cords

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
        self.n_samples = list_files
        
            
    
    def __len__(self):
        return len(self.n_samples)
    
    def __getitem__(self,idx):
        try:
            if torch.is_tensor(idx):
                idx = idx.tolist()
            image_name = self.n_samples[idx]

            original_image_path = os.path.join(self.original_root,image_name)
            toon_image_path = os.path.join(self.toon_root,image_name)

            original_image = np.asarray(Image.open(original_image_path).convert('RGB'))
            toon_image = np.asarray(Image.open(toon_image_path).convert('RGB'))

            # print(original_image.shape, toon_image.shape)

            ############################ Extracting Faces from Images ###################
            detections = config.DETECTOR.detect(original_image)
            xmin, ymin, xmax, ymax, c = detections[0]
            height = ymax-ymin
            width = xmax-xmin
                    #------ To add forehead and chin uisng height and extend width in order to fit the full face -------#
            xmin_mod_1, ymin_mod_1, xmax_mod_1, ymax_mod_1 = int(xmin - width/6), int(ymin - height/4), int(xmax + width/6), int(ymax + width/15)
                    #------ Checking Cordinates to keep it within valid ranges [0,480] ------#
            xmin_mod_1, ymin_mod_1, xmax_mod_1, ymax_mod_1 = check_cords(xmin_mod_1, ymin_mod_1, xmax_mod_1, ymax_mod_1)
            
            real_face_image = original_image[ymin_mod_1:ymax_mod_1, xmin_mod_1:xmax_mod_1]
            toon_face_image = toon_image[ymin_mod_1:ymax_mod_1, xmin_mod_1:xmax_mod_1]


            augmentations = both_transform(image=real_face_image, image0=toon_face_image)
            input_image = augmentations["image"]
            target_image = augmentations["image0"]

            real_face_image = transform_only_input(image=input_image)["image"]
            toon_face_image = transform_only_mask(image=target_image)["image"]

            return (real_face_image, toon_face_image)
        except:
            if torch.is_tensor(idx):
                idx = idx.tolist()
            image_name = self.n_samples[idx]
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
        try:
            if torch.is_tensor(idx):
                idx = idx.tolist()
            image_name = self.n_samples[idx]

            image_path = os.path.join(self.root,image_name)

            original_image = np.asarray(Image.open(image_path).convert('RGB'))
            ################### Extracting Face #######################
            detection = config.DETECTOR.detect(original_image)
            xmin, ymin, xmax, ymax, c = detection[0]
            height = ymax-ymin
            width = xmax-xmin
            xmin_mod_1, ymin_mod_1, xmax_mod_1, ymax_mod_1 = int(xmin - width/6), int(ymin - height/4), int(xmax + width/6), int(ymax + width/15)

            face_image = original_image[ymin_mod_1:ymax_mod_1, xmin_mod_1:xmax_mod_1]


            if self.transform:
                face_image = self.transform(face_image)
            else:
                face_image = transform_test_data(face_image)

            return face_image
        except:
            if torch.is_tensor(idx):
                idx = idx.tolist()
            image_name = self.n_samples[idx]
            image_path = os.path.join(self.root,image_name)
            print(image_path)
            pass
    
            
if __name__=="__main__":
    dataset = Face_Toon_Dataset("./toon_pix_data/train/")
    loader = DataLoader(dataset, batch_size=1)
    count = 0
    for x,y in loader:
        print("X Shape :-",x.shape)
        print("Y Shape :-",y.shape)
        count +=1
    print(count)
    #     # save_image(x*0.5 + 0.5,"original.png")
    #     # save_image(y*0.5 + 0.5,"toon.png") 
    # dataset = Test_Faces("./toon_pix_data/test/")
    # loader = DataLoader(dataset, batch_size=1)
    # count = 0
    # for x in loader:
    #     print(x.shape)
                  