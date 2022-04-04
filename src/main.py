import argparse
import imp
import parser
import logging
import sys
import inspect
import os 
import shutil
import json  
import ast 

# Add paths 
sys.path.append(os.path.abspath(os.path.join('..')))

import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 

import torch 
import torchvision 
import torch.distributed as dist
import torch.optim as optim
from torchvision import transforms, utils
torch.cuda.empty_cache()

from torch.utils.data import Dataset, DataLoader
import torch.utils.data.distributed 
from PIL import Image

import dataset_
import utility_
import model_


# set Sagemaker config only for local machine 
os.environ["SM_HOSTS"] = "['algo1','algo2']"
os.environ["SM_CURRENT_HOST"] = "algo2"
os.environ["SM_MODEL_DIR"] = "../models/AWS/"
os.environ["SM_CHANNEL_TRAINING"] = "/opt/ml/input/data/train"
os.environ["SM_NUM_GPUS"] = '0'


# logging 
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(filename='example.log', level=logging.DEBUG, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

logger.addHandler(logging.StreamHandler(sys.stdout))


def _get_train_data_loader(batch_size, train_Dataset, is_distributed, **kwargs):
    
    logger.info("Get train data loader")
     
    train_sampler = (torch.utils.data.distributed.DistributedSampler(train_Dataset) if is_distributed else None)

    return torch.utils.data.DataLoader(train_Dataset,
                                      batch_size=batch_size, 
                                      shuffle = train_sampler is None,
                                      sampler = train_sampler,
                                      **kwargs)

def _get_test_data_loader(test_batch_size, test_Dataset, **kwargs):

    logger.info("Get test data loader")
    
    return torch.utils.data.DataLoader(test_Dataset,
                                      batch_size=test_batch_size, 
                                      shuffle = True,
                                      **kwargs)

def _average_gradients(model):
    
    # Gradient Averaging 
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data/=size


def train_loop(train_loader,model,age_criterion,gender_criterion,
                optimizer,is_distributed,use_cuda,device):

    size = len(train_loader.dataset)
    model.train()
    
    for batch,(img,age,gender) in enumerate(train_loader):
        
        img, age, gender = img.to(device), age.to(device), gender.to(device)
        
        # compute prediction error 
        pred = model(img)
        
        age_loss = age_criterion(pred[0].to(device),age.long())
        gender_loss = gender_criterion(pred[1].to(device),gender.long())

        loss = (age_loss + gender_loss)/ 2 

        # Backprop 
        optimizer.zero_grad()
        loss.backward()

        if is_distributed and not use_cuda:
                # average gradients manually for multi-machine cpu case only 
                _average_gradients(model)

        optimizer.step()

        if batch % args.log_interval == 0:
            age_loss,gender_loss,current = age_loss.item(),gender_loss.item(), batch*len(img)
            print(f"Age loss: {age_loss:>7f} Gender loss: {gender_loss:>7f} [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model,valid_loss_min_input,optimizer,age_criterion,gender_criterion,epoch,
            checkpoint_path,best_model_path,device):


    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()

    # initialize tracker for minimum validation loss
    valid_loss_min = valid_loss_min_input 

    test_age_loss,test_gender_loss, correct_age,correct_gender = 0,0,0,0

    for batch,(img,age,gender) in enumerate(dataloader):

        img,age,gender = img.to(device), age.to(device), gender.to(device)
        pred = model(img)
    
        test_age_loss += age_criterion(pred[0].to(device), age.long())
        test_gender_loss += gender_criterion(pred[1].to(device), gender.long())

        correct_age += (pred[0].to(device).argmax(1) == age).type(torch.float).sum().item()
        correct_gender += (pred[1].to(device).argmax(1) == gender).type(torch.float).sum().item()
        
    test_age_loss/=num_batches
    test_gender_loss /= num_batches
    correct_age /= size
    correct_gender /= size

    print(f"Test Error \n Age Accuracy: {100*correct_age:>2f} Gender Accuracy: {100*correct_gender:>2f} \n Age loss: {test_age_loss:>7f} Gender loss: {test_gender_loss:>7f}")

    combined_loss = (test_age_loss + test_gender_loss)/ 2 

    # create checkpoint variable and add important data
    checkpoint = {
        'epoch': epoch + 1,
        'valid_loss':combined_loss,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        }
        
    # save checkpoint   
    utility_.save_checkpoint(checkpoint, False, checkpoint_path, best_model_path)
        
    if combined_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,combined_loss))
        # save checkpoint as best model
        utility_.save_checkpoint(checkpoint, True, checkpoint_path, best_model_path)
        valid_loss_min = combined_loss


def train(args):

    # Check for distributed training
    is_distributed = len(args.hosts) > 1 and args.backend is not None
    
    logger.debug("Distributed Training - {}".format(is_distributed))
    use_cuda = args.num_gpus > 0
    
    logger.debug("Number of gpus available - {}".format(args.num_gpus))
    kwargs = {"num_workers":1, "pin_memory":True} if use_cuda else {}
    
    device = torch.device("cuda" if use_cuda else "cpu")
   
    if is_distributed:
        
        # Initalize the distributed environment. 
        world_size = len(args.hosts)
        os.environ["WORLD_SIZE"] = str(world_size)
        host_rank = args.hosts.index(args.current_host)
        os.environ["RANK"] = str(host_rank)
        # dist.init_process_group(backend=args.backend, rank=host_rank, world_size=world_size)
        # logger.info(
            # "Initialized the distributed environment: '{}' backend on {} nodes.".format(args.backend, dist.get_world_size())
            #  + "Current host rank is {}. Number of gpus: {}".format(dist.get_rank(),args.num_gpus))
    

    # Set the seed for generating random numbers 
    torch.manual_seed(args.seed)

    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    # Create a dataset from Image files and json 
    Age_Gender_Dataset = dataset_.Gender_Age_Classifier_dataset(json_file=os.path.join(args.data_dir,'data_dict.json'), root_dir=os.path.join(args.data_dir,'aligned'),
                                                                transforms=transforms.Compose([transforms.ToTensor(),transforms.Resize((104,104))]))

    lengths = [int(len(Age_Gender_Dataset)*0.8), len(Age_Gender_Dataset)-int(len(Age_Gender_Dataset)*0.8)]
    train_Dataset, val_Dataset = torch.utils.data.random_split(Age_Gender_Dataset, lengths)

    train_loader = _get_train_data_loader(args.batch_size, train_Dataset, is_distributed, **kwargs)
    test_loader = _get_test_data_loader(args.test_batch_size, val_Dataset, **kwargs)

    logger.debug("Processes {}/{} ({:.0f}%) of train data".format(len(train_loader.sampler),len(train_loader.dataset),
                                                                 100.0*len(train_loader.sampler)/len(train_loader.dataset),))

    logger.debug("Processes {}/{} ({:.0f}%) of test data".format(len(test_loader.sampler),len(test_loader.dataset),
                                                                100.0 * len(test_loader.sampler) / len(test_loader.dataset),))
    
    # Gets all the classes in the model_ file 
    cls_members = inspect.getmembers(model_, inspect.isclass)

    for class_name,class_obj in cls_members:
        
        if class_name == args.architecture:
            model = class_obj().to(device)
            model_name = class_name
            logger.debug("Model architecture - {}".format(model_name))

    if is_distributed and use_cuda:
        
        # multi-machine multi-gpu case 
        model = torch.nn.parallel.DistributedDataParallel(model)
    
    else:
        
        # Single-machine multi-gpu case or single-machine or mult-machine cpu case 
        model = torch.nn.DataParallel(model)


    # Loss function
    gender_criterion = torch.nn.CrossEntropyLoss()
    age_criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
    
    checkpoint_folder = os.path.join(args.model_dir,"checkpoint")
    best_model_folder = os.path.join(args.model_dir,"best_model")

    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)
    
    if not os.path.exists(best_model_folder):
        os.makedirs(best_model_folder)
        
    checkpoint_path = os.path.join(checkpoint_folder,"{}_checkpoint.pt".format(model_name))
    best_model_path = os.path.join(best_model_folder,"{}_best_model.pt".format(model_name))


    valid_loss_min_input = np.Inf

    for i in range(args.epochs):
        print(f'Epoch:{i+1}\n ------------------------------')
        
        train_loop(train_loader,model,age_criterion,gender_criterion,optimizer,
                    is_distributed,use_cuda,device)

        test_loop(test_loader, model,valid_loss_min_input,optimizer,age_criterion,
                    gender_criterion,i,checkpoint_path,best_model_path,device)

    

    




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="PyTorch Age + Gender Classifier")
    

    # Data and model checkpoints directories
    parser.add_argument("--batch-size",type=int,default=64,metavar="N",
                        help="input batch size for training (default: 64)",)
    
    parser.add_argument("--test-batch-size",type=int,default=1000,metavar="N",
                        help="input batch size for testing (default: 1000)",)
    
    parser.add_argument("--epochs",type=int,default=10,metavar="N",
                        help="number of epochs to train (default: 10)",)
    
    parser.add_argument("--lr", type=float, default=0.001, 
                        metavar="LR", help="learning rate (default: 0.001)")
    
    parser.add_argument("--momentum", type=float, default=0.5, 
                        metavar="M", help="SGD momentum (default: 0.5)")
    
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    
    parser.add_argument("--log-interval", type=int,default=100,metavar="N",
                        help="how many batches to wait before logging training status",)
    
    parser.add_argument("--backend", type=str,default=None,
                        help="backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)",)

    parser.add_argument("--architecture", type=str, default='Base_CNN_multi_task',
                        help="Architecture to use for training (default: Base_CNN_multi_task)",)

    # Container environment
    parser.add_argument("--hosts", type=list, default=ast.literal_eval(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])

    args = parser.parse_args()
    print(args)

    train(args)




