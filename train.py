# train.py
# --------
# censing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational purposes.# project. You are free to use and extend these projects for educational purposes.

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
import datetime
import random
import re
import pprint
import argparse
from util import cal_iter_time, hamming_score
from pytz import timezone
from torch.utils.data import DataLoader
from dataset import BassetDataset
from model import ResNet1d, ResNet2d, Basset
from sklearn.metrics import accuracy_score

tz = timezone('US/Eastern')
pp = pprint.PrettyPrinter(indent=4)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', type=str, default='./data', help="Path to the dataset directory (default: './data')")
parser.add_argument('--file_name', type=str, default='sample_dataset.h5', help='Name of the h5 dataset file already preprocessed in the preprocessing step (default: sample_dataset.h5)')
parser.add_argument('--log_dir', default='log/', help='Base log folder (create if it does not exist')
parser.add_argument('--log_name', default='basset_train', help='name to use when logging this model')
parser.add_argument('--network_type', default='resnet1d', help="Which type of model architecture to use ('basset', 'resnet1d', 'resnet2d', etc)")
parser.add_argument('--batch_size', type=int, default=64, help='Defines the batch size for training phase (default: 64)')
parser.add_argument('--nb_epochs', type=int, default=2000, help='Defines the maximum number of epochs the network needs to train (default: 200)')
parser.add_argument('--optimizer', type=str, default='adam', help="The algorithm used for the optimization of the model (default: 'adam')")
parser.add_argument('--learning_rate', type=float, default=0.002, help='Learning rate for the optimizer (default: 0.004)')
parser.add_argument('--beta1', type=float, default=0.9, help="'beta1' for the optimizer")
parser.add_argument('--seed', type=int, default=313, help='Seed for reproducibility')
args = parser.parse_args()

# some assertions
assert args.network_type in ['basset', 'resnet1d', 'resnet2d'], "The input '{}' as the network type is not implemented!".format(args.network_type)


# sets device for model and PyTorch tensors
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set RNG
seed = args.seed
torch.backends.cudnn.deterministic = True
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if device.type=='cuda':
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# save args in the log folder

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("Arguments are:")
pp.pprint(vars(args))
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

# sets device for model and PyTorch tensors
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set RNG
seed = args.seed
torch.backends.cudnn.deterministic = True
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if device.type=='cuda':
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# create 'log_dir' folder (if it does not exist already)
os.makedirs(args.log_dir, exist_ok=True)

# building the dataloaders needed
basset_dataset_train = BassetDataset(path=args.dataset_dir, f5name=args.file_name, split='train')
basset_dataset_valid = BassetDataset(path=args.dataset_dir, f5name=args.file_name, split='valid')

# using default pytorch DataLoaders
basset_dataloader_train = DataLoader(basset_dataset_train, batch_size=args.batch_size, drop_last=True, shuffle=True, num_workers=8)
basset_dataloader_valid = DataLoader(basset_dataset_valid, batch_size=args.batch_size, drop_last=False, shuffle=False, num_workers=8)

# basset network instantiation
if args.network_type=="basset":
    classifier = Basset().to(device)
elif args.network_type=="resnet1d":
    classifier = ResNet1d().to(device)
elif args.network_type=="resnet2d":
    classifier = ResNet2d().to(device)

# cost function
criterion = nn.BCEWithLogitsLoss()

# setup optimizer & scheduler
if args.optimizer=='adam':
    optimizer = optim.Adam(list(classifier.parameters()), lr=args.learning_rate, betas=(args.beta1, 0.999))
elif args.optimizer=='rmsprop':
    optimizer = optim.RMSprop(list(classifier.parameters()), lr=args.learning_rate)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)  # use an exponentially decaying learning rate

# keeping track of the time
start_time = datetime.datetime.now(tz)
former_iteration_endpoint = start_time
print("~~~~~~~~~~~~~ TIME ~~~~~~~~~~~~~~")
print("Time started: {}".format(start_time.strftime("%Y-%m-%d %H:%M:%S")))
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

# main training loop
for n_epoch in range(args.nb_epochs):
    # training data
    #train_preds = np.empty((0, 164))
    #train_targets = np.empty((0, 164))
    #train_loss = []
    for n_batch, batch_samples in enumerate(basset_dataloader_train):

        optimizer.zero_grad()
        seqs, trgs = batch_samples[0], batch_samples[1]
        predictions = classifier(seqs.reshape(seqs.shape[0], 4, 600).float().to(device))
        loss = criterion(predictions, trgs.float().to(device))
        loss.backward()
        optimizer.step()

        #train_loss.append(criterion(predictions, trgs.float().to(device)).item())
        #train_preds = np.concatenate((train_preds, torch.sigmoid(predictions).detach().cpu().numpy()), axis=0)
        #train_targets = np.concatenate((train_targets, trgs), axis=0)
    
    # validation data
    valid_preds = np.empty((0, 164))
    valid_targets = np.empty((0, 164))
    valid_loss = []
    for n_batch, batch_samples in enumerate(basset_dataloader_valid):

        seqs, trgs = batch_samples[0], batch_samples[1]
        predictions = classifier.eval()(seqs.reshape(seqs.shape[0], 4, 600).float().to(device))

        valid_loss.append(criterion(predictions, trgs.float().to(device)).item())
        valid_preds = np.concatenate((valid_preds, torch.sigmoid(predictions).detach().cpu().numpy()), axis=0)
        valid_targets = np.concatenate((valid_targets, trgs), axis=0)

    former_iteration_endpoint, time_elapsed = cal_iter_time(former_iteration_endpoint, tz)
    #train_acc = hamming_score(train_targets > 0.5, train_preds > 0.5, normalize=True, sample_weight=None)
    valid_acc = hamming_score(valid_targets > 0.5, valid_preds > 0.5, normalize=True, sample_weight=None)
    print("Epoch {}/{}: valid_loss={:.3f}, valid_accuracy={:.2f}% ({} hh:mm:ss)".format(
        n_epoch+1, args.nb_epochs, np.sum(valid_loss), valid_acc*100, time_elapsed))
    #print("Epoch {}/{}: train_loss={:.3f}, train_accuracy={:.2f}%, valid_loss={:.3f}, valid_accuracy={:.2f}% ({} hh:mm:ss)".format(
    #    n_epoch+1, args.nb_epochs, np.sum(train_loss), train_acc*100, np.sum(valid_loss), valid_acc*100, time_elapsed))
            
    # show/save stats of the results in the log folder
    # checkpoint the basset_net in the log folder
    scheduler.step()

