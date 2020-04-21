# train.py
# --------
# censing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational purposes.# project. You are free to use and extend these projects for educational purposes.

import os
from os import path
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
import requests
import \
    logomaker  # https://github.com/jbkinney/logomaker/tree/master/logomaker/tutorials (should be moved to the util.py)
import argparse
from util import cal_iter_time, hamming_score
from pytz import timezone
from torch.utils.data import DataLoader
from dataset import RecombDataset
from model import ResNet1d, ResNet2d, Recomb
from sklearn.metrics import accuracy_score
from tqdm import tqdm

tz = timezone('US/Eastern')
pp = pprint.PrettyPrinter(indent=4)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', type=str, default='./data',
                    help="Path to the dataset directory (default: './data')")
parser.add_argument('--file_name', type=str, default='sample_dataset.h5',
                    help='Name of the h5 dataset file already preprocessed in the preprocessing step (default: sample_dataset.h5)')
parser.add_argument('--log_dir', default='log/', help='Base log folder (create if it does not exist')
parser.add_argument('--log_name', default='recomb_train', help='name to use when logging this model')
parser.add_argument('--network_type', default='resnet1d',
                    help="Which type of model architecture to use ('recomb', 'resnet1d', 'resnet2d', etc)")
parser.add_argument('--batch_size', type=int, default=64,
                    help='Defines the batch size for training phase (default: 64)')
parser.add_argument('--nb_epochs', type=int, default=200,
                    help='Defines the maximum number of epochs the network needs to train (default: 200)')
parser.add_argument('--optimizer', type=str, default='adam',
                    help="The algorithm used for the optimization of the model (default: 'adam')")
parser.add_argument('--validate', type=bool, default=True, help='Whether to use validation set')
parser.add_argument('--learning_rate', type=float, default=0.004,
                    help='Learning rate for the optimizer (default: 0.004)')
parser.add_argument('--beta1', type=float, default=0.5, help="'beta1' for the optimizer")
parser.add_argument('--seed', type=int, default=313, help='Seed for reproducibility')
args = parser.parse_args()

# some assertions
assert args.network_type in ['recomb', 'resnet1d',
                             'resnet2d'], "The input '{}' as the network type is not implemented!".format(
    args.network_type)

# sets device for model and PyTorch tensors
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set RNG
seed = args.seed
torch.backends.cudnn.deterministic = True
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if device.type == 'cuda':
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# save args in the log folder

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("Arguments are:")
pp.pprint(vars(args))
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

# sets device for model and PyTorch tensors
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set RNG
seed = args.seed
torch.backends.cudnn.deterministic = True
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if device.type == 'cuda':
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# create 'log_dir' folder (if it does not exist already)
os.makedirs(args.log_dir, exist_ok=True)

# building the dataloaders needed
recomb_dataset_train = RecombDataset(path=args.dataset_dir, f5name=args.file_name, split='train')
recomb_dataset_valid = RecombDataset(path=args.dataset_dir, f5name=args.file_name, split='valid')

# using default pytorch DataLoaders
recomb_dataloader_train = DataLoader(recomb_dataset_train, batch_size=args.batch_size, drop_last=True, shuffle=True,
                                     num_workers=8)
recomb_dataloader_valid = DataLoader(recomb_dataset_valid, batch_size=args.batch_size, drop_last=False, shuffle=False,
                                     num_workers=8)

# recomb network instantiation
if args.network_type == "recomb":
    classifier = Recomb().to(device)
elif args.network_type == "resnet1d":
    classifier = ResNet1d().to(device)
elif args.network_type == "resnet2d":
    classifier = ResNet2d().to(device)

# cost function
criterion = nn.BCEWithLogitsLoss()

# setup optimizer & scheduler
if args.optimizer == 'adam':
    optimizer = optim.Adam(list(classifier.parameters()), lr=args.learning_rate, betas=(args.beta1, 0.999))
elif args.optimizaer == 'rmsprop':
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
    t = tqdm(recomb_dataloader_train, ncols=160, desc="Epoch {}/{}".format(n_epoch + 1, args.nb_epochs))
    for n_batch, batch_samples in enumerate(t):
        optimizer.zero_grad()
        # if 10 < n_batch: break
        seqs, trgs = batch_samples[0], batch_samples[1]
        predictions = classifier(seqs.reshape(args.batch_size, 4, 600).float().to(device))
        loss = criterion(predictions, trgs.float().to(device))
        loss.backward()
        optimizer.step()
        acc = hamming_score(trgs.float() > 0.5, torch.sigmoid(predictions).detach().cpu().numpy() > 0.5, normalize=True,
                            sample_weight=None)
        t.set_postfix(train_loss="{:.3f}".format(loss.item()), train_accuracy="{:.2f}%".format(acc * 100))

        # validation
        if n_batch == len(recomb_dataloader_train) - 1 and args.validate:
            preds = np.empty((0, 164))
            targets = np.empty((0, 164))
            valid_loss = []
            for n_batch, batch_samples in enumerate(recomb_dataloader_valid):
                seqs, trgs = batch_samples[0], batch_samples[1]
                predictions = classifier.eval()(seqs.reshape(seqs.shape[0], 4, 600).float().to(device))
                valid_loss.append(criterion(predictions, trgs.float().to(device)).item())
                preds = np.concatenate((preds, torch.sigmoid(predictions).detach().cpu().numpy()), axis=0)
                targets = np.concatenate((targets, trgs), axis=0)
            # predictions = np.argmax(preds > 0.5, axis=1)
            # trgs = np.argmax((targets > 0.5), axis=1)
            # print("targets", (targets > 0.5)[0]*1)
            # print("preds", (preds > 0.5)[0]*1)
            acc = hamming_score(targets > 0.5, preds > 0.5, normalize=True, sample_weight=None)
            t.set_postfix(valid_loss="{:.3f}".format(np.sum(valid_loss)), valid_accuracy="{:.2f}%".format(acc * 100))
            # t.set_postfix(valid_accuracy="{}".format(acc))

    # show/save stats of the results in the log folder
    # checkpoint the recomb_net in the log folder
    # former_iteration_endpoint = cal_iter_time(former_iteration_endpoint, tz)
    scheduler.step()

