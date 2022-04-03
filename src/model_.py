
import torch 
from torch import nn
import torch.nn.functional as F
 
# Base cnn with two heads 
class Base_CNN_multi_task(nn.Module):
    def __init__(self):
        """
        CNN model with 2 heads 
        """
        super(Base_CNN_multi_task,self).__init__()
        
        self.conv1 = nn.Conv2d(3,6,3)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,10,3)
        self.pool2 = nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(10,32,2)
        self.pool3 = nn.MaxPool2d(2,2)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(3872, 1000)
        self.fc2 = nn.Linear(1000, 100)       

        self.head1 = nn.Linear(100,8)
        self.head2 = nn.Linear(100,3)
    
    def forward(self,x):
        
        x = F.relu(self.conv1(x))
        # print(x.shape)
        x = self.pool1(x)
        # print(x.shape)

        x = F.relu(self.conv2(x))
        # print(x.shape)
        x = self.pool2(x)
        # print(x.shape)

        x = F.relu(self.conv3(x))
        # print(x.shape)
        x = self.pool3(x)
        # print(x.shape)

        x = self.flatten(x)
        # print(x.shape)
        
        x = self.fc1(x)
        # print(x.shape)

        x = self.fc2(x)
        # print(x.shape)

        # head 1 age 
        x1 = self.head1(x)
        # print(x1.shape)
        
        # head 2 gender
        x2 = self.head2(x)
        # print(x2.shape)

        return x1,x2


class Resnet_multi_task(nn.Module):

    def __init__(self):
        """
        Resnet pretrained network 

        """

        super(Resnet_multi_task,self).__init__()
        
        self.resnet = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
        
        self.resnet.fc = nn.Linear(512,100)
        
        self.head1 = nn.Linear(100,8)
        self.head2 = nn.Linear(100,3)


    def forward(self,x):
        
        x = self.resnet(x)
        # print(x.shape)

        # head 1 age 
        x1 = self.head1(x)
    
        # head 2 gender
        x2 = self.head2(x)

        return x1,x2

class MobileNet_multi_task(nn.Module):

    def __init__(self):
        """
        MobileNet pretrained network 

        """

        super(MobileNet_multi_task,self).__init__()
        
        self.mobilenet = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)
        
        self.mobilenet.classifier[1] = nn.Linear(1280,100)
        
        self.head1 = nn.Linear(100,8)
        self.head2 = nn.Linear(100,3)

    def forward(self,x):
        
        x = self.mobilenet(x)
        # print(x.shape)

        # head 1 age 
        x1 = self.head1(x)
    
        # head 2 gender
        x2 = self.head2(x)

        return x1,x2
        






