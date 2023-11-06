# Audio activity detection based on signal eveleop
from icecream import ic
from matplotlib import pyplot as plt
import torch
import torchaudio

# audio, sr = torchaudio.load("/Users/kyeongmukkang/Documents/GitHub/DATA/FSD50K/test/Hammer/127995.wav")
# audio, sr = torchaudio.load("/Users/kyeongmukkang/Documents/GitHub/DATA/FSD50K/test/Walk_and_footsteps/272381.wav")
# audio, sr = torchaudio.load("/Users/kyeongmukkang/Documents/GitHub/DATA/FSD50K/test/Train/396982.wav")
# audio, sr = torchaudio.load("/Users/kyeongmukkang/Documents/GitHub/DATA/FSD50K/test/Organ/78436.wav")
audio, sr = torchaudio.load("/Users/kyeongmukkang/Documents/GitHub/DATA/FSD50K/test/Laughter/427699.wav")
thresh = 0.1

def aad(audio, thresh,):
    audio_envelop = abs(audio)
    audio_max = torch.max(audio_envelop)
    audio_activity_detection = torch.zeros(audio.shape)
    
    idx = audio_envelop > audio_max*thresh
    audio_activity_detection[idx] = 1

    content_percentage = len(audio_activity_detection[idx])/len(audio_activity_detection.squeeze()) * 100 

    return audio_activity_detection, content_percentage


# plt.plot(aad(audio, thresh,)[0].squeeze())
# plt.plot(audio.squeeze())
# plt.show()
# ic(aad(audio, thresh,)[1])