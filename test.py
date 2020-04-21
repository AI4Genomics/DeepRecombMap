# test.py
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
import requests
import logomaker # https://github.com/jbkinney/logomaker/tree/master/logomaker/tutorials (should be moved to the util.py)
import argparse
from pytz import timezone

basset_dataset_test = BassetDataset(path='./data/', f5name='sample_dataset.h5', split='test')
print("The number of samples in {} split is {}.".format('test', len(basset_dataset_test)))
print("The first 10 ids of test samples are:\n  {}\n".format("\n  ".join(basset_dataset_test.ids[:10])))