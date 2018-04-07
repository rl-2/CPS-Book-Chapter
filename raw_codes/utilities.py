import math
import numpy as np
import torchvision.transforms as transforms
from IPython.core.debugger import Pdb
import platform
if platform.system() == "Windows":
    import msvcrt
from PIL import Image
import torch

def out_size(Hin, Win, kernel_size, stride=1, padding=0, dilation=1):
    Hout = math.floor((Hin+2*padding-dilation*(kernel_size-1)-1)/stride+1)
    Wout = math.floor((Win+2*padding-dilation*(kernel_size-1)-1)/stride+1)    
    return(Hout, Wout)

def processImage(obs):
    #Pdb().set_trace()
    #obs = Image.fromarray(obs)

    #normalize = transforms.Normalize(mean=[.53], std=[.21])

    #preprocess = transforms.Compose([
    #    transforms.ToTensor(),
    #    normalize
    #])
    obs = (obs-.53)/.21
    obs = np.expand_dims(obs,0)
    obs = torch.Tensor(obs)
    #obs = preprocess(obs)
    obs = obs.unsqueeze(0)
    return(obs)

# For getting numbers for normilization prior to training and running the DNN
def calculate_mean_std(all_obs):
    """all_obs is a list of numpy tensors (height x width x channels)"""
    total_mean = np.zeros_like(np.mean(np.mean(all_obs[0], axis=0), axis=0))
    total_std = np.zeros_like(np.std(np.std(all_obs[0], axis=0), axis=0))
    N = len(all_obs)
    for obs in all_obs:
        obs = obs/256
        total_mean += np.mean(np.mean(obs, axis=0), axis=0)
        total_std  += np.std(np.std(obs, axis=0), axis=0)
    mean = total_mean/N
    std = total_std/N
    return mean, std

def convert_numpy_to_grey(rgb):
    """ 
    input: rgb scaled between [0,1]
    output: greyscale
    """
    def weightedAverage(pixel):
        return 0.299*pixel[0] + 0.587*pixel[1] + 0.114*pixel[2]

    grey = np.zeros((rgb.shape[0], rgb.shape[1])) # init 2D numpy array
    for rownum in range(len(rgb)):
        for colnum in range(len(rgb[rownum])):
            grey[rownum][colnum] = weightedAverage(rgb[rownum][colnum])
    return grey

# asks whether a key has been acquired
def kbfunc():
    #this is boolean for whether the keyboard has been hit
    x = msvcrt.kbhit()
    if x:
        #getch acquires the character encoded in binary ASCII
        ret = msvcrt.getch()
    else:
        ret = False
    return ret    