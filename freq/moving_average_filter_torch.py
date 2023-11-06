from icecream import ic
from matplotlib import pyplot as plt
import torch
import torchaudio
from icecream import ic
import torch.nn.functional as F
import torch.nn as nn

audio1, sr = torchaudio.load("/Users/kyeongmukkang/Documents/GitHub/DATA/FSD50K/test/Hammer/127995.wav")

def MAF(audio, kernel_size=5, padding='same'):
    weights = 1/kernel_size * torch.ones(kernel_size).unsqueeze(0).unsqueeze(0) #RuntimeError: weight should have at least three dimensions
    weights.requires_grad = False

    # MA = nn.Conv1d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    MA = nn.Conv1d(1, 1, kernel_size=kernel_size, stride=1, padding=padding, bias=False)

    with torch.no_grad():
        MA.weight = nn.Parameter(weights)
        # ic(MA.weight)

    output = MA(audio)
    return output

# test_intput = torch.ones([1,30])
# outputs = MAF(test_intput)
# ic(outputs)