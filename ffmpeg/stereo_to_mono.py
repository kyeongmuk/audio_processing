import subprocess
import os
from icecream import ic
from pathlib import Path


# def stereo_to_mono(input_file, output_L_file, output_R_file):
#     ffmpeg_cmd = [
#         "ffmpeg",
#         "-i", input_file,
#         "-filter_complex","[0:a]pan=1c|c0=c0[left];[0:a]pan=1c|c0=c1[right]",
#         "-map", "[left]", output_L_file, "-map", "[right]", output_R_file,
#     ]
    
#     try:
#         subprocess.run(ffmpeg_cmd, check=True)
#         print("Succesfully converted")
#     except subprocess.CalledProcessError as e:
#         print("Conversion failed!")
    
    
def stereo_to_mono(input_file, output_L_file, output_R_file):
    ffmpeg_cmd = [
        "ffmpeg",
        "-i", input_file,
        "-map_channel", "0.0.0", output_L_file,
        "-map_channel", "0.0.1", output_R_file,
    ]
    
    try:
        subprocess.run(ffmpeg_cmd, check=True)
        print("Succesfully converted")
    except subprocess.CalledProcessError as e:
        print("Conversion failed!")
    
    
    
    
folders = f"/Volumes/kkm_T7_4/DB/MUSDB18/musdb18hq/test"
# exts = ['flac', 'wav', 'mp3']
exts = ['wav']
tar_files = []

path = Path(folders)
tar_files = [str(file) for ext in exts for file in path.glob(f'**/*.{ext}')]


for i, filename in enumerate(tar_files):
    ic(i, filename)
    if filename.endswith("wav"):
        input_file = filename
        output_file_L = (f"{filename}".split(".wav")[0] + "_L.wav").replace("test", "test_wav_split")
        output_file_R = (f"{filename}".split(".wav")[0] + "_R.wav").replace("test", "test_wav_split")
        os.makedirs(os.path.dirname(output_file_L), exist_ok=True)
        stereo_to_mono(filename, output_file_L, output_file_R)
