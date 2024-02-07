import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

import os
from icecream import ic
from pathlib import Path


# paths = [f"/Volumes/kkm_T7_4/DB/Valentini/Valentini_48k/clean_trainset_56spk_wav", "/Volumes/kkm_T7_4/DB/Valentini/Valentini_48k/clean_trainset_28spk_wav", "/Volumes/kkm_T7_4/DB/Valentini/Valentini_48k/clean_testset_wav", f"/Volumes/kkm_T7_4/DB/Valentini/Valentini_48k/noisy_trainset_56spk_wav", "/Volumes/kkm_T7_4/DB/Valentini/Valentini_48k/noisy_trainset_28spk_wav", "/Volumes/kkm_T7_4/DB/Valentini/Valentini_48k/noisy_testset_wav"]
paths = ["/Volumes/kkm_T7_4/DB/Valentini/Valentini_48k"]
# exts = ['flac', 'wav', 'mp3']

exts = ['wav']
all_files = []

for p in paths:
    path = Path(p)
    files = [str(file) for ext in exts for file in path.glob(f'**/*.{ext}')]
    all_files.append(files)
    
ic(all_files)

resample_rate = 24000

for file in files:
    waveform, sample_rate = torchaudio.load(file)
    resampler = T.Resample(sample_rate, resample_rate, dtype=waveform.dtype)
    resampled_waveform = resampler(waveform)
    
    # ic(file.replace('Valentini_48k', 'Valentini_16k'))
    # ic(os.path.dirname(file.replace('Valentini_48k', 'Valentini_16k')))
    os.makedirs(os.path.dirname(file.replace('Valentini_48k', 'Valentini_24k')), exist_ok=True)
    torchaudio.save(file.replace('Valentini_48k', 'Valentini_24k'), resampled_waveform, sample_rate=resample_rate, format="wav", bits_per_sample=16)