# model.py
# --------
# censing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational purposes.# project. You are free to use and extend these projects for educational purposes.

import numpy as np
import torch
import torch.nn as nn
import argparse
import torch.nn.functional as F

class Basset(nn.Module):
    def __init__(self):
        super(Basset, self).__init__()

        self.dropout = 0.3
        self.num_cell_types = 164

        self.conv1 = nn.Conv2d(4, 300, (19, 1), stride = (1, 1), padding=(9,0))
        self.conv2 = nn.Conv2d(300, 200, (11, 1), stride = (1, 1), padding = (5,0))
        self.conv3 = nn.Conv2d(200, 200, (7, 1), stride = (1, 1), padding = (4,0))


        self.bn1 = nn.BatchNorm2d(300)
        self.bn2 = nn.BatchNorm2d(200)
        self.bn3 = nn.BatchNorm2d(200)
        self.maxpool1 = nn.MaxPool2d((3, 1))
        self.maxpool2 = nn.MaxPool2d((4, 1))
        self.maxpool3 = nn.MaxPool2d((4, 1))

        self.fc1 = nn.Linear(13*200, 1000)
        self.bn4 = nn.BatchNorm1d(1000)

        self.fc2 = nn.Linear(1000, 1000)
        self.bn5 = nn.BatchNorm1d(1000)

        self.fc3 = nn.Linear(1000, self.num_cell_types)

    def forward(self, s):
        #s = s.permute(0, 2, 1).contiguous()                          # batch_size x 4 x 600
        s = s.view(-1, 4, 600, 1)                                   # batch_size x 4 x 600 x 1 [4 channels]
        s = self.maxpool1(F.relu(self.bn1(self.conv1(s))))           # batch_size x 300 x 200 x 1
        s = self.maxpool2(F.relu(self.bn2(self.conv2(s))))           # batch_size x 200 x 50 x 1
        s = self.maxpool3(F.relu(self.bn3(self.conv3(s))))           # batch_size x 200 x 13 x 1
        s = s.view(-1, 13*200)
        conv_out = s

        s = F.dropout(F.relu(self.bn4(self.fc1(s))), p=self.dropout, training=self.training)  # batch_size x 1000
        s = F.dropout(F.relu(self.bn5(self.fc2(s))), p=self.dropout, training=self.training)  # batch_size x 1000

        s = self.fc3(s)

        return s#, conv_out

   
class resblock_1d(nn.Module):
    """Class to return a block of 1D Resudual Networks
    
    Parameters:
        inputs (tensor):Input of the previous layer
        num_channels (int):Dimension parameter (in each resnet block).
        kernel_size (int):Size of the (1D) filters (or kernels).

    Returns:
        outputs (torch.nn.Module):'resblock_1d' module that consist of several "conv1d->BN->relu" repetitions
    
    """
    def __init__(self, num_channels, kernel_size=5):
        super(resblock_1d, self).__init__()
        modules = []
        relu = nn.ReLU(True)
        for r in range(2):
            modules.append(
                nn.Sequential(
                    nn.Conv1d(num_channels, num_channels, kernel_size, stride=1, padding=kernel_size//2),  # (batch, width, out_chan)
                    nn.BatchNorm1d(num_channels),
                    relu
                )
            )
        self.block = nn.Sequential(*modules)
        
    def forward(self, inputs):
        outputs = self.block(inputs)
        return outputs


class ResNet1d(nn.Module):
    """ Resudual Network with 1D convolutions for discriminating real vs generated sequences.

    Parameters:
        inputs (tensor):Tensor of size (batch_size, vocab_size, max_seq_len) containing real or generated sequences.
        num_channels (int):Discriminator dimension parameter (in each resnet block).
        vocab_size (int):Size of the first layer input channel.
        seq_len (int):Length of the input sequence.
        num_layers (int):How many repetitions of 'resblock_1d' for discriminator.

    Returns:
        outputs (tensor):Batch of (single) values for real or generated inputs.

    """
    def __init__(self, num_channels=64, vocab_size=4, seq_len=600, num_classes=164, res_layers=2):
        super(ResNet1d, self).__init__()
        
        self.num_channels = num_channels
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.res_layers = res_layers
        self.num_classes = num_classes
                
        self.conv = nn.Sequential(
                nn.Conv1d(vocab_size, num_channels, 1, stride=1, padding=0)  # bottleneck: (batch, width, out_chan)
            )
        resblocks = []
        for i in range(res_layers):
            resblocks.append(resblock_1d(self.num_channels))
        self.resblocks = nn.Sequential(*resblocks)
        self.prediction_layer = nn.Linear(seq_len*num_channels, num_classes)
            
    def forward(self, inputs):
        outputs = self.conv(inputs)
        inputs = outputs
        for i in range(self.res_layers):
            outputs = 1.0*self.resblocks[i](inputs) + inputs  # where resnet idea comes into play!
            inputs = outputs
        outputs = torch.reshape(outputs, [-1, self.seq_len*self.num_channels])
        outputs = self.prediction_layer(outputs)
        return outputs


class resblock_2d(nn.Module):
    """Class to return a block of 2D Resudual Networks
    
    Parameters:
        inputs (tensor):Input of the previous layer
        num_channels (int):Dimension parameter (in each resnet block).
        kernel_size (tuple):Size of the (2D) filters (or kernels).

    Returns:
        outputs (torch.nn.Module):'resblock_2d' module that consist of several "conv2d->BN->relu" repetitions
    
    """
    def __init__(self, num_channels, kernel_size=(5, 3)):
        super(resblock_2d, self).__init__()
        modules = []
        relu = nn.ReLU(True)
        for r in range(2):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(num_channels, num_channels, kernel_size, stride=(1, 1), 
                              padding=(kernel_size[0]//2, kernel_size[1]//2)),  # bottleneck: (batch, width, out_chan)
                    nn.BatchNorm2d(num_channels),
                    relu
                )
            )
        self.block = nn.Sequential(*modules)
    
    def forward(self, inputs):
        outputs = self.block(inputs)
        return outputs


class ResNet2d(nn.Module):
    """ Resudual Network with 2D convolutions for discriminating real vs generated sequences.
    
    Parameters:
        inputs (tensor):Tensor of size (batch_size, vocab_size, max_seq_len) containing real or generated sequences.
        num_channels (int):Discriminator dimension parameter (in each resnet block).
        vocab_size (int):Size of the first layer input channel.
        seq_len (int):Length of the input sequence.
        num_layers (int):How many repetitions of 'resblock_2d' for discriminator.

    Returns:
        outputs (tensor):Batch of 2D tensors of values in the size (vocab_size, seq_len).   

    """
    def __init__(self, num_channels=128, vocab_size=4, seq_len=600, num_classes=164, res_layers=2):
        super(ResNet2d, self).__init__()
        
        self.num_channels = num_channels
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.res_layers = res_layers
        self.num_classes = num_classes
                
        self.conv = nn.Sequential(
                nn.Conv2d(1, num_channels, (3, 1), stride=(1, 1), padding=(3//2, 0))  # (batch, width, out_chan)
            )
        resblocks = []
        for i in range(res_layers):
            resblocks.append(resblock_2d(self.num_channels))
        self.resblocks = nn.Sequential(*resblocks)
        self.prediction_layer = nn.Linear(self.vocab_size*seq_len*num_channels, num_classes)
            
    def forward(self, inputs):
        inputs = torch.reshape(inputs, [-1, 1, self.vocab_size, self.seq_len])
        outputs = self.conv(inputs)
        inputs = outputs
        for i in range(self.res_layers):
            outputs = 1.0*self.resblocks[i](inputs) + inputs  # where resnet idea comes into play!
            inputs = outputs
        outputs = torch.reshape(outputs, [-1, self.vocab_size*self.seq_len*self.num_channels])
        
        outputs = self.prediction_layer(outputs)
        return outputs


#net = ResNet1d() # __init__ here
#random_sample = torch.tensor(np.random.randn(64, 4, 1, 600)).float()  # 64 is the batch_size
#net(random_sample) # forward here
