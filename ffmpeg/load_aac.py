import librosa
from icecream import ic

data, sr = librosa.load("/Volumes/kkm_T7_4/DB/MUSDB18/musdb18hq/train_aac/Actions - Devil's Words/mixture.m4a", sr=44100)
ic(data, sr)