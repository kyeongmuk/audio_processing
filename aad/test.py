from data import *
import torch
import torchaudio
from matplotlib import pyplot as plt
from icecream import ic
import itertools
from silence_detection.aad_envelop_based import aad
from einops import pack, unpack, rearrange
import numpy as np
from natsort import natsorted
import pandas as pd
import random

def cycle(dl):
    while True:
        for data in dl:
            yield data
            
folders = ["/Users/kyeongmukkang/Documents/GitHub/DATA/FSD50K/test"]#, "/Users/kyeongmukkang/Documents/GitHub/DATA/VCTK/test"]

ds = SoundDataset(folders=folders, target_sample_hz=48000, max_length=48000*4)
# ic(ds.__len__())

test_loader = get_dataloader(ds, shuffle=True)
# test_loader = get_dataloader(ds, shuffle=False)
# ic(test_loader.dataset[0])
data_iter = cycle(test_loader)

data = []
data_aad = []
data_percent = []
data_length = []

num_data = 32
thresh = 0.1

idx_normal = []
idx_abnormal = []

for i in range(num_data):
    x = next(data_iter)[0]
    x_aad, x_percent = aad(x, thresh)
    
    data.append(np.array(x.squeeze()))
    data_aad.append(np.array(x_aad))
    data_percent.append(x_percent)
    data_length.append(len(x.squeeze()))
     
# ic(data, data_aad, data_percent)
df = pd.DataFrame({
    'data': data,
    'length': data_length,
    'aad': data_aad,
    'percent': data_percent,
    'portion': None,
})

df = df.sort_values(by=['percent'], ascending=True)
df = df.reset_index(drop=True)

for i in range(num_data):
    percent = df['percent'][i]
    if percent < 40: idx_normal.append(i)
    else: idx_abnormal.append(i)
# ic(len(idx_normal), len(idx_abnormal))

if len(idx_normal) % 4 != 0:
    # ic(len(idx_normal) % 4)
    res = 4 - len(idx_normal) % 4
    idx_normal.extend(idx_abnormal[:res])
    # idx_abnormal.remove[:res]
    idx_abnormal = idx_abnormal[res:]

# ic(len(idx_normal), len(idx_abnormal))

max_length = max(df['length'])
idx = 0
split_num = 5


for idx in range(len(df)):
    portions = []
    max_num_portion = df['length'][idx]//split_num
    for i in range(split_num):
        portions.append(np.sum(df['aad'][idx][:, max_num_portion*(i):max_num_portion*(i+1)]))
    
    df['portion'][idx] = np.argsort(portions)
    

df_normal = df.loc[idx_normal].sort_values(by=['percent'], ascending=True)
df_abnormal = df.loc[idx_abnormal].sort_values(by=['percent'], ascending=True)
# ic(idx_normal, idx_abnormal)
num_group = int(len(df_normal) / 4)
# ic(num_group)

def grouping_1(idx_normal, num_group):
    groups = []
    for i in range(num_group):    
        groups.append([idx_normal[-(i+1)], idx_normal[num_group*0 + i], idx_normal[num_group*1 + i], idx_normal[num_group*2 + i]])
    return groups


groups = grouping_1(idx_normal, num_group)

max_len = max(df['length'][groups[0]])
shuffle_order = np.arange(0, 3)
def pad_groups(df, groups, max_len, padding_val = 0):
    for g in range(len(groups)):
        max_len = max(df['length'][groups[g]])
        np.random.permutation(shuffle_order)
        for i in range(len(df['data'][groups[g]])):
            # ic(i)
            max_padding = max_len-df['length'][groups[g][i]]
            # ic(pad_num)
            # if pad_num == 0 : continue
            # cnt = shuffle_order[i]
            # ic(i%4)
            
            pad_left = random.randint(0, max_padding)
            # ic(pad_left)
            pad_right = int(max_padding - pad_left)
            # ic(max_padding, start+end, start, end)
            # ic(max_padding, max_len, df['length'][groups[g][i]])

            df['data'][groups[g][i]] = np.pad(df['data'][groups[g][i]], pad_width=(pad_left, pad_right))

            df['length'][groups[g][i]] = len(df['data'][groups[g][i]])
            
pad_groups(df_normal, groups, max_len)     

# ic(df_normal)
# ic(idx_normal)
# ic(idx_abnormal)
# ic(groups)
stack_data = []
prepared_data = []
for g in range(len(groups)):
    stack_data = []
    for i in range(len(df_normal['data'][groups[g]])):
        # ic(torch.from_numpy(df_normal['data'][groups[g]].values[i]).shape)
        stack_data.append(torch.from_numpy(df_normal['data'][groups[g]].values[i]))
    
    # ic(g, i)
    prepared_data.append(torch.stack(stack_data))
    
# ic(df)
    
for i in range(len(groups)):
    torchaudio.save(f'./test/4object_test{i}.wav', prepared_data[i], sample_rate=48000, format='wav', bits_per_sample=16)
    
    
for i in range(len(idx_abnormal)):
    torchaudio.save(f'./test/4object_abnormal_test{i}.wav', torch.from_numpy(df_abnormal['data'].values[i]).unsqueeze(0), sample_rate=48000, format='wav', bits_per_sample=16)
    