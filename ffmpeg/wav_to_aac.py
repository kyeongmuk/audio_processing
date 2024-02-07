import subprocess
import os
from icecream import ic
from pathlib import Path

def convert_wav_to_aac(input_file, output_file):
    ffmpeg_cmd = [
        "ffmpeg",
        "-i", input_file,
        "-c:a", "aac",
        "-b:a", "128k",
        output_file        
    ]
    
    try:
        subprocess.run(ffmpeg_cmd, check=True)
        print("Succesfully converted")
    except subprocess.CalledProcessError as e:
        print("Conversion failed!")
    
    
    
    
folders = f"/Volumes/kkm_T7_4/DB/MUSDB18/musdb18hq/test_wav_split"
exts = ['flac', 'wav', 'mp3']
tar_files = []

path = Path(folders)
save_path = "/data"
ic(path)
tar_files = [str(file) for ext in exts for file in path.glob(f'**/*.{ext}')]


for i, filename in enumerate(tar_files):
    ic(i, filename)
    if filename.endswith("wav"):
        input_file = filename
        output_file = (f"{filename}".split(".wav")[0] + ".m4a").replace("test_wav_split", "test_wav_split_to_aac128")
        # ic(i,(f"{filename}".split("wav")[0] + "m4a"), output_file)

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        convert_wav_to_aac(filename, output_file)
