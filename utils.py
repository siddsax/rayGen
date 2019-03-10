import argparse
import os
import numpy as np
import math
import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
import PIL
# from PIL import Image
from matplotlib import cm
class rayleighData(torch.utils.data.Dataset):

  def __init__(self, dataPath, transform=None, train=True, val=False):
        'Initialization'
        self.dataPath = dataPath
        self.transform = transform
        self.train = train
        self.filenames = open(dataPath + '/fileNames.txt').read().split('\n')

        # self.labelDict = {
        #     '1e6' : torch.from_numpy(np.array([1, 0, 0]).reshape(1, 3)),
        #     '2e6' : torch.from_numpy(np.array([.7, .3, 0]).reshape(1, 3)),
        #     '5e6' : torch.from_numpy(np.array([.31, .69, 0]).reshape(1, 3)), 
        #     '1e7' : torch.from_numpy(np.array([0, 1, 0]).reshape(1, 3)),
        #     '2e7' : torch.from_numpy(np.array([0, .7, .3]).reshape(1, 3)),
        #     '5e7' : torch.from_numpy(np.array([0, .31, .69]).reshape(1, 3)),
        #     '1e8' : torch.from_numpy(np.array([0, 0, 1]).reshape(1, 3))
        # }
        # [1e6, 2e6, 5e6, 1e7, 2e7, 5e7, 1e8]
        # self.labelDict = {
        #     '1e6' : torch.from_numpy(np.array([3, 0, 0]).reshape(1, 3)),
        #     '2e6' : torch.from_numpy(np.array([2, 1, 0]).reshape(1, 3)),
        #     '5e6' : torch.from_numpy(np.array([1, 2, 0]).reshape(1, 3)), 
        #     '1e7' : torch.from_numpy(np.array([0, 3, 0]).reshape(1, 3)),
        #     '2e7' : torch.from_numpy(np.array([0, 2, 1]).reshape(1, 3)),
        #     '5e7' : torch.from_numpy(np.array([0, 1, 2]).reshape(1, 3)),
        #     '1e8' : torch.from_numpy(np.array([0, 0, 3]).reshape(1, 3))
        # }

        # self.labelDict = {
        #     '1e6' : torch.from_numpy(np.array([1, 0, 0, 0, 0, 0, 0])),
        #     '2e6' : torch.from_numpy(np.array([0, 1, 0, 0, 0, 0, 0])),
        #     '5e6' : torch.from_numpy(np.array([0, 0, 1, 0, 0, 0, 0])), 
        #     '1e7' : torch.from_numpy(np.array([0, 0, 0, 1, 0, 0, 0])),
        #     '2e7' : torch.from_numpy(np.array([0, 0, 0, 0, 1, 0, 0])),
        #     '5e7' : torch.from_numpy(np.array([0, 0, 0, 0, 0, 1, 0])),
        #     '1e8' : torch.from_numpy(np.array([0, 0, 0, 0, 0, 0, 1]))
        # }

        self.labelDict = {
            '1e6' : 0,
            '2e6' : 1,
            '5e6' : 2, 
            '1e7' : 3,
            '2e7' : 4,
            '5e7' : 5,
            '1e8' : 6
        }

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.filenames)

  def __getitem__(self, index):

        while(1):
            try:
                fN = self.filenames[index%len(self.filenames)]
                x = np.load(self.dataPath + '/' + fN)
                y = self.labelDict[fN[:3]]
                break
            except:
                index+=1
                print(fN)
                continue

        #try:
        x = PIL.Image.fromarray(x)
        #except:
        #    import pdb;pdb.set_trace()
        if self.transform:
            x = self.transform(x)

        return x, y
