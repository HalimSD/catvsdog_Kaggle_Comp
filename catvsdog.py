# RUN THE FOLLOWING IN THE CLI
'''
! mkdir data
! unzip /kaggle/input/dogs-vs-cats/train.zip -d /kaggle/working/data
! unzip /kaggle/input/dogs-vs-cats/test1.zip -d /kaggle/working/data
! mv /kaggle/working/data/test1 /kaggle/working/data/test
!cp /kaggle/input/dogs-vs-cats/sampleSubmission.csv /kaggle/working/data
'''

import os
from PIL import Image as PILImage
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as transforms
import numpy as np
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dir = '/kaggle/working/data/train'
test_dir = '/kaggle/working/data/test'
train_files = os.listdir(train_dir)
test_files = os.listdir(test_dir)
cat_files = [cat_file for cat_file in train_files if 'cat' in cat_file]
dog_files = [dog_file for dog_file in train_files if 'dog' in dog_file]



class MyDataSet(Dataset):
    def __init__(self, root_dir, files_list, mode='train', transform=None):
        self.transform = transform
        self.root_dir = root_dir
        self.files_list = files_list
        self.mode = mode
        if mode == 'train':
            if 'dog' in files_list[0]:
                self.label = 1
            else:
                self.label = 0
        
    def __len__(self):
        return len(self.files_list)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.root_dir, self.files_list[index])
        image = PILImage.open(image_path)
        if self.transform:
            image = self.transform(image)
        if self.mode == 'train':
            image = np.array(image)
            return image.astype('float32') , self.label
        else:
            image = np.array(image)
            return image.astype('float32'), self.files_list[index]
        
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225], inplace = True)
])

cats_dataset = MyDataSet(train_dir, cat_files, transform = transform)
dogs_dataset = MyDataSet(train_dir, dog_files, transform = transform)
dogs_cats_dataset = ConcatDataset([cats_dataset, dogs_dataset])
train_set, val_set = torch.utils.data.random_split(dogs_cats_dataset, [22000, 3000])
train_loader = DataLoader(dataset = train_set, batch_size = 32, shuffle = True, drop_last= True)
val_loader = DataLoader(dataset = val_set, batch_size = 32, shuffle = True, drop_last= True)

model = torchvision.models.googlenet(pretrained = True)
model.to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr = 1e-3)

model.train()
for epoch in range(3):   
    losses = []
    for batch_index, (image,label) in enumerate(train_loader):
        image = image.to(device)
        label = label.to(device)
        
        scores = model(image)
        loss = criterion(scores, label)
        losses.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f"cost at epoch {epoch} is {sum(losses)/len(losses)}")

torch.save(model.state_dict(), '/kaggle/working/ckpt_net_catdog.pt')
# model.load_state_dict(torch.load('/kaggle/working/data/ckpt_net_catdog.pt'))


# model evaluation on the validation set
with torch.no_grad():
    accuracy_list = []
    test_loss = 0
    correct = 0
    for data, target in val_loader:
        output = model(data)
        test_loss += criterion(output, target).item()                                                              
        pred = output.data.max(1, keepdim=True)[1]                                                                
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

    test_loss /= len(val_loader.dataset)
    accuracy = 100. * correct / len(val_loader.dataset)
    accuracy_list.append(accuracy)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(val_loader.dataset),accuracy))



# model testing

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225], inplace = True)
])

testset = MyDataSet(test_dir, test_files, mode='test', transform = test_transform)
testloader = DataLoader(testset, batch_size = 32, shuffle=False)


model.eval()
fn_list = []
pred_list = []
for image, label in testloader:
    with torch.no_grad():
        image = image.to(device)
        output = model(image)
        pred = torch.argmax(output, dim=1)
        fn_list += [n[:-4] for n in label]
        pred_list += [p.item() for p in pred]

submission = pd.DataFrame({"id":fn_list, "label":pred_list})
submission.to_csv('prediction_googlenet.csv', index=False)



