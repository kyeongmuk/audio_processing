from icecream import ic
from matplotlib import pyplot as plt
import torch
import torchaudio
from icecream import ic
import torch.nn.functional as F
import torch.nn as nn

audio1, sr = torchaudio.load("/Users/kyeongmukkang/Documents/GitHub/DATA/FSD50K/test/Hammer/127995.wav")

#audio shape: [B, T]
def MAXF(audio, kernel_size=5):
    weights = torch.ones(kernel_size) #RuntimeError: weight should have at least three dimensions
    output = torch.zeros(audio.shape)
    for i in range(audio.shape[1] - (kernel_size-1)):
        output[..., i] = torch.max(audio[..., i:(i+kernel_size)])

    for i in range(audio.shape[1] - (kernel_size-1), audio.shape[1]):
        output[..., i] = torch.max(audio[..., i:])

    return output

# test_intput = torch.ones([2,30])
# outputs = MEDF(test_intput)
# ic(outputs)
# ic(outputs.shape)