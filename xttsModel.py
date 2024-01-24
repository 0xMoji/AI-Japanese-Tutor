import torch
from TTS.api import TTS
from playsound import playsound

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"
# List available 🐸TTS models
print(TTS().list_models())

# Init TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def xttsV2(response):
    tts.tts_to_file(
        text = response, 
        speaker_wav="新垣結衣さん.mp3", 
        language="ja", 
        file_path="speech.mp3"
        )
    playsound("speech.mp3")