import torch
import json
import os 
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils 
from torchvision.io import read_image

class Gender_Age_Classifier_dataset(Dataset):
    "Dataset for the Age Gender Classifier"
    
    def __init__(self,root_dir,json_file,transforms=None):
        """
        Args:
            root_dir (string):
            json_file (string):
            transform (): 
        """
        self.root_dir = root_dir
        self.data = json.load(open(json_file))[0]
        self.transform = transforms
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):

        image_name = list(self.data.keys())[idx]
        image_path = os.path.join(self.root_dir,image_name)
        image = Image.open(image_path)
        image = image.convert('RGB')
        image = np.asarray(image)
    
        age = torch.tensor(self.data[image_name]['age_id'])
        gender = torch.tensor(self.data[image_name]['gender_id'])

        if self.transform:
            image = self.transform(image)
            
        return image, age, gender

















