from icecream import ic
from matplotlib import pyplot as plt
import torch
import torchaudio
from aad_envelop_based import aad
import torch.nn.functional as F

audio1, sr = torchaudio.load("/Users/kyeongmukkang/Documents/GitHub/DATA/FSD50K/test/Hammer/127995.wav")
audio2, sr = torchaudio.load("/Users/kyeongmukkang/Documents/GitHub/DATA/FSD50K/test/Walk_and_footsteps/272381.wav")
audio3, sr = torchaudio.load("/Users/kyeongmukkang/Documents/GitHub/DATA/FSD50K/test/Train/396982.wav")
audio4, sr = torchaudio.load("/Users/kyeongmukkang/Documents/GitHub/DATA/FSD50K/test/Organ/78436.wav")
audio5, sr = torchaudio.load("/Users/kyeongmukkang/Documents/GitHub/DATA/FSD50K/test/Laughter/427699.wav")

thresh = 0.1

aad1, aad1_percent = aad(audio1, thresh)
aad2, aad2_percent = aad(audio2, thresh)
aad3, aad3_percent = aad(audio3, thresh)
aad4, aad4_percent = aad(audio4, thresh)
aad5, aad5_percent = aad(audio5, thresh)
