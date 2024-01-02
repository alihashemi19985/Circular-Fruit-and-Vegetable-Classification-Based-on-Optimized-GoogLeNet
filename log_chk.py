import pandas as pd 
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn.init as init
import torch.nn as nn
from tqdm import tqdm
from torchvision import utils,datasets
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, Dataset, DataLoader
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingWarmRestarts
import logging
import os 


logging.basicConfig(filename="newfile1.log",
                        format='%(asctime)s %(message)s',
                        filemode='w')

    # Creating an object
logger = logging.getLogger()

    # Setting the threshold of logger to DEBUG
logger.setLevel(logging.DEBUG)

logger.info("Just an information")

class Checkpoint(object):
    def __init__(self):
        self.best_acc = 0.
        self.folder = 'chekpoint'
        os.makedirs(self.folder, exist_ok=True)
    def save(self, acc, filename, epoch,net):
        if acc > self.best_acc:
            logger.info('Saving checkpoint...')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            path = os.path.join(os.path.abspath(self.folder), filename + '.pth')
            torch.save(state, path)
            self.best_acc = acc
    def load(self,clf):
        W = torch.load(r'C:\Users\Ali\Desktop\Deep Learning papers\Fruit_detection\chekpoint\chk.pth')['net']
        clf.load_state_dict(W)
        return clf

       