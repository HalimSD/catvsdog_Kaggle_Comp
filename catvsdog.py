# RUN THE FOLLOWING IN THE CLI
'''
! mkdir data
! unzip /kaggle/input/dogs-vs-cats/train.zip -d /kaggle/working/data
! unzip /kaggle/input/dogs-vs-cats/test1.zip -d /kaggle/working/data
! mv /kaggle/working/data/test1 /kaggle/working/data/test
!cp /kaggle/input/dogs-vs-cats/sampleSubmission.csv /kaggle/working/data
'''

import os
import csv
import re as re
from glob import glob
from PIL import Image
import pandas as pd
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.preprocessing import LabelBinarizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

classes = set()
file_names = []
images = []
labels = []
ids = []

trainning_files = glob("/kaggle/working/data/train/*.jpg")
for path in trainning_files:
    file_full_name = re.findall(r'\w+\.\d+.jpg', path)[0]
    id = re.findall(r'.(\d+)\.', path)[0]
    name = re.findall(r'/(\w+)\.', path)[0]
    classes.add(name)
    img = Image.open(path)
    images.append(img)
    labels.append(name)
    ids.append(id)
    file_names.append(file_full_name)

data_w_images = []
data_w_images = list(zip(images, labels, ids, file_names))
data = []
data = list(zip(file_names, ids, labels))

with open('./data/data.csv', 'wt', newline='') as csvfile: 
    header = ['file_name', 'id', 'label']
    csvwriter = csv.writer(csvfile, delimiter=',')
    csvwriter.writerow(('file_name', 'id', 'label'))
    csvwriter.writerows(data)
csvfile.close()

class MyDataSet(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.transform = transform
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = Image.open(image_path)
        label = self.annotations.iloc[index, 2]
        
        if self.transform:
            image = self.transform(image)
            
        return (image, label)

data_set = MyDataSet(csv_file = './data/data.csv', root_dir = 'data/train', transform = transforms.Compose([
    (lambda y: torch.zeros(2, dtype=torch.float32).scatter_(0, torch.tensor(y), value=1).long()),
    transforms.ToTensor()]))

train_set, val_set = torch.utils.data.random_split(data_set, [22000, 3000])

train_loader = DataLoader(dataset = train_set, batch_size = 32, shuffle = True, drop_last= True)
val_loader = DataLoader(dataset = val_set, batch_size = 32, shuffle = True, drop_last= True)


