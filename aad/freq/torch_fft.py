from icecream import ic
from matplotlib import pyplot as plt
import torch
import torchaudio
from icecream import ic
import torch.nn.functional as F
import torch.nn as nn
from moving_average_filter_torch import MAF
from median_filter_torch import MEDF
from maximum_filter_torch import MAXF

audio1, sr = torchaudio.load("/Users/kyeongmukkang/Documents/GitHub/DATA/FSD50K/test/Hammer/127995.wav")
audio2, sr = torchaudio.load("/Users/kyeongmukkang/Documents/GitHub/DATA/FSD50K/test/Walk_and_footsteps/272381.wav")
audio3, sr = torchaudio.load("/Users/kyeongmukkang/Documents/GitHub/DATA/FSD50K/test/Train/396982.wav")
audio4, sr = torchaudio.load("/Users/kyeongmukkang/Documents/GitHub/DATA/FSD50K/test/Organ/78436.wav")
audio5, sr = torchaudio.load("/Users/kyeongmukkang/Documents/GitHub/DATA/FSD50K/test/Laughter/427699.wav")

# torchaudio.save("./audio1.wav", audio1, sample_rate = sr)
# torchaudio.save("./audio2.wav", audio2, sample_rate = sr)
# torchaudio.save("./audio3.wav", audio3, sample_rate = sr)
# torchaudio.save("./audio4.wav", audio4, sample_rate = sr)
# torchaudio.save("./audio5.wav", audio5, sample_rate = sr)

audio1_fft = torch.fft.rfft(audio1)
x = torch.linspace(0,48000,len(audio1_fft.squeeze()))
ic(x)
ic(len(x))
audio2_fft = torch.fft.rfft(audio2)
audio3_fft = torch.fft.rfft(audio3)
audio4_fft = torch.fft.rfft(audio4)
audio5_fft = torch.fft.rfft(audio5)

max_16bit = torch.mul(20, torch.log10(torch.tensor(2**16)))
dB_audio1_fft = (torch.mul(10,torch.log10(audio1_fft.real**2 + audio1_fft.imag**2)) - max_16bit).squeeze()
dB_audio2_fft = (torch.mul(10,torch.log10(audio2_fft.real**2 + audio2_fft.imag**2)) - max_16bit).squeeze()
dB_audio3_fft = (torch.mul(10,torch.log10(audio3_fft.real**2 + audio3_fft.imag**2)) - max_16bit).squeeze()
dB_audio4_fft = (torch.mul(10,torch.log10(audio4_fft.real**2 + audio4_fft.imag**2)) - max_16bit).squeeze()
dB_audio5_fft = (torch.mul(10,torch.log10(audio5_fft.real**2 + audio5_fft.imag**2)) - max_16bit).squeeze()

kernel_size = 100
padding = 0
MAF_dB1 = MAF(dB_audio1_fft.unsqueeze(0), kernel_size = kernel_size, padding=padding)
MAF_dB2 = MAF(dB_audio2_fft.unsqueeze(0), kernel_size = kernel_size, padding=padding)
MAF_dB3 = MAF(dB_audio3_fft.unsqueeze(0), kernel_size = kernel_size, padding=padding)
MAF_dB4 = MAF(dB_audio4_fft.unsqueeze(0), kernel_size = kernel_size, padding=padding)
MAF_dB5 = MAF(dB_audio5_fft.unsqueeze(0), kernel_size = kernel_size, padding=padding)

# kernel_size = 30
padding = 0
MEDF_dB1 = MEDF(dB_audio1_fft.unsqueeze(0), kernel_size = kernel_size)
MEDF_dB2 = MEDF(dB_audio2_fft.unsqueeze(0), kernel_size = kernel_size)
MEDF_dB3 = MEDF(dB_audio3_fft.unsqueeze(0), kernel_size = kernel_size)
MEDF_dB4 = MEDF(dB_audio4_fft.unsqueeze(0), kernel_size = kernel_size)
MEDF_dB5 = MEDF(dB_audio5_fft.unsqueeze(0), kernel_size = kernel_size)

# kernel_size = 30
padding = 0
MAXF_dB1 = MAXF(dB_audio1_fft.unsqueeze(0), kernel_size = kernel_size)
MAXF_dB2 = MAXF(dB_audio2_fft.unsqueeze(0), kernel_size = kernel_size)
MAXF_dB3 = MAXF(dB_audio3_fft.unsqueeze(0), kernel_size = kernel_size)
MAXF_dB4 = MAXF(dB_audio4_fft.unsqueeze(0), kernel_size = kernel_size)
MAXF_dB5 = MAXF(dB_audio5_fft.unsqueeze(0), kernel_size = kernel_size)

ic(len(MEDF_dB1.squeeze()))
ic(MEDF_dB1.squeeze())
plt.plot(torch.linspace(0, sr//2, len(dB_audio1_fft)), dB_audio1_fft.squeeze())
# plt.plot(torch.linspace(0, sr//2, len(dB_audio1_fft)), MAF(dB_audio1_fft.unsqueeze(0), kernel_size = 100).squeeze().detach().numpy())
# plt.show()
plt.plot(torch.linspace(0, sr//2, len(MAF_dB1.squeeze())), MAF_dB1.squeeze().detach().numpy())
plt.plot(torch.linspace(0, sr//2, len(MEDF_dB1.squeeze())), MEDF_dB1.squeeze().detach().numpy())
plt.plot(torch.linspace(0, sr//2, len(MAXF_dB1.squeeze())), MAXF_dB1.squeeze().detach().numpy())
# plt.plot(torch.linspace(0, sr//2, len(dB_audio1_fft)), moving_avg_filter(dB_audio1_fft.unsqueeze(0)).squeeze().detach().numpy())
plt.show()

plt.plot(torch.linspace(0, sr//2, len(dB_audio2_fft)), dB_audio2_fft)
plt.plot(torch.linspace(0, sr//2, len(MAF_dB2.squeeze())), MAF_dB2.squeeze().detach().numpy())
plt.plot(torch.linspace(0, sr//2, len(MEDF_dB2.squeeze())), MEDF_dB2.squeeze().detach().numpy())
plt.plot(torch.linspace(0, sr//2, len(MAXF_dB2.squeeze())), MAXF_dB2.squeeze().detach().numpy())
plt.show()

plt.plot(torch.linspace(0, sr//2, len(dB_audio3_fft)), dB_audio3_fft)
plt.plot(torch.linspace(0, sr//2, len(MAF_dB3.squeeze())), MAF_dB3.squeeze().detach().numpy())
plt.plot(torch.linspace(0, sr//2, len(MEDF_dB3.squeeze())), MEDF_dB3.squeeze().detach().numpy())
plt.plot(torch.linspace(0, sr//2, len(MAXF_dB3.squeeze())), MAXF_dB3.squeeze().detach().numpy())
plt.show()

plt.plot(torch.linspace(0, sr//2, len(dB_audio4_fft)), dB_audio4_fft)
plt.plot(torch.linspace(0, sr//2, len(MAF_dB4.squeeze())), MAF_dB4.squeeze().detach().numpy())
plt.plot(torch.linspace(0, sr//2, len(MEDF_dB4.squeeze())), MEDF_dB4.squeeze().detach().numpy())
plt.plot(torch.linspace(0, sr//2, len(MAXF_dB4.squeeze())), MAXF_dB4.squeeze().detach().numpy())
plt.show()

plt.plot(torch.linspace(0, sr//2, len(dB_audio5_fft)), dB_audio5_fft)
plt.plot(torch.linspace(0, sr//2, len(MAF_dB5.squeeze())), MAF_dB5.squeeze().detach().numpy())
plt.plot(torch.linspace(0, sr//2, len(MEDF_dB5.squeeze())), MEDF_dB5.squeeze().detach().numpy())
plt.plot(torch.linspace(0, sr//2, len(MAXF_dB5.squeeze())), MAXF_dB5.squeeze().detach().numpy())
plt.show()