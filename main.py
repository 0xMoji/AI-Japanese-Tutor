from pathlib import Path
import openai
from pydub import AudioSegment
from pydub.playback import play
from openai import OpenAI
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
from scipy.io.wavfile import write
import json
from playsound import playsound
import torch
from TTS.api import TTS
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

def xttsV2_streaming(response, char_audio):
    # Load model configuration and model
    config = XttsConfig()
    config.load_json("/path/to/xtts/config.json")
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir="/path/to/xtts/", use_deepspeed=True)
    model.cuda()

    # Get speaker latents
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[char_audio])

    # Generate audio chunks
    chunks = model.inference_stream(
        response,
        "ja",
        gpt_cond_latent,
        speaker_embedding
    )

    # Here you need to decide how to return these chunks. One option is to save them as individual files
    # and return a list of file paths. Another is to concatenate them and save as one file.
    wav_chunks = [chunk for chunk in chunks]
    
    # Concatenate and save (or handle differently based on your application's needs)
    wav = torch.cat(wav_chunks, dim=0)
    torchaudio.save("audio/output_streamed.wav", wav.squeeze().unsqueeze(0).cpu(), 24000)

    return "audio/output_streamed.wav"



# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"
# Init TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def xttsV2(response, char_audio):
    tts.tts_to_file(
        text = response, 
        speaker_wav = char_audio, 
        language = "ja", 
        file_path = "audio/output.wav"
        )
    # playsound("output.wav")

client = OpenAI(api_key="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")

# Define trigger words
trigger_words = ["再见","拜拜"]  # Add more words as needed

# Updated gpt function to include conversation history
def gpt_convo(text, conversation_history, actress_name):
    conversation_history.append({"role": "user", "content": f"""
    Pretend you are the famous Japanese actress {actress_name} talking to your loved one. 
    Reply to the following Japanese conversation dialogue:{text}. 
    If you found any grammar mistakes in the above dialogue, respond in Japanese to teach your lover how to correctly say it. 
    Only respond in Japanese(make sure you do not provide hiragana in this section) followed by Hiragana, Romanized Pinyin, and translation in Chinese in JSON format. The following is a formatting example.
    {{"Japanese":"","Hiragana":"","Romanized Pinyin":"","Chinese":""}}
    """})
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=conversation_history
    )
    response_text = completion.choices[0].message.content
    response_dict = json.loads(response_text)
    japanese, hiragana, pinyin, chinese = response_dict['Japanese'], response_dict['Hiragana'], response_dict['Romanized Pinyin'], response_dict['Chinese']
    print(f"""GPT:\n"Japanese":{japanese}\n"Hiragana":{hiragana}\n"Romanized Pinyin":{pinyin}\n"Chinese":{chinese}""")
    conversation_history.append({"role": "assistant", "content": response_text})
    print(conversation_history)
    return response_dict

# Updated gpt function to include conversation history
def gpt(text, conversation_history):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": f""""
        Translate the following Chinese into Japanese:{text}. 
        Only respond in Japanese followed by Hiragana, Romanized Pinyin, and translation in Chinese in JSON format. The following is a formatting example.
        {{"Japanese":"","Hiragana":"","Romanized Pinyin":"","Chinese":""}}
        """}]
    )
    response_text = completion.choices[0].message.content

    response_dict = json.loads(response_text)
    # response_dict = response_dict.decode('utf-8')
    japanese, hiragana, pinyin, chinese = response_dict['Japanese'], response_dict['Hiragana'], response_dict['Romanized Pinyin'], response_dict['Chinese']
    print(f"""GPT:\n"Japanese":{japanese}\n"Hiragana":{hiragana}\n"Romanized Pinyin":{pinyin}\n"Chinese":{chinese}""")
    return response_dict

def tts_model(text):
    speech_file_path = Path(__file__).parent / "audio/speech.mp3"
    response = client.audio.speech.create(
        model="tts-1",
        voice="echo",
        input=f"{text}",
        response_format="mp3"
    )
    response.stream_to_file(speech_file_path)
    playsound("audio/speech.mp3")

def stt_model(file, lang):
    audio_file = open(file, "rb")
    transcript = client.audio.transcriptions.create(
        model="whisper-1", 
        file=audio_file,
        language=lang
    )
    print("User:\n" + transcript.text)
    return transcript.text

def record_audio(filename, fs=48000, threshold=0.01, silence_duration=1.5):
# def record_audio(filename, fs=24000, threshold=0.01, silence_duration=1.5):
    """
    Record audio from the microphone and save it to a file, stopping the recording when silence is detected for more than 1 second.

    Parameters:
    filename : str
        The name of the file where to store the recording.
    fs : int, optional
        The sampling rate of the recording (default is 44100 Hz).
    threshold : float, optional
        The sound level below which is considered silence (default is 0.01).
    silence_duration : int, optional
        Duration of silence in seconds before stopping the recording (default is 1 second).
    """
    print("⭕️Recording...")
    with sd.InputStream(samplerate=48000, channels=1, device=1) as stream:
    # with sd.InputStream(samplerate=24000, channels=1, device=0) as stream:
        audio_data = []
        num_silent_frames = 0
        silent_frames_to_stop = fs * silence_duration

        while True:
            data, _ = stream.read(fs)
            audio_data.append(data)
            if np.abs(data).mean() < threshold:
                num_silent_frames += len(data)
            else:
                num_silent_frames = 0

            if num_silent_frames >= silent_frames_to_stop:
                break

    audio_data = np.concatenate(audio_data)
    write(filename, fs, audio_data)
    print(f"Recording saved as {filename}")

# Function to check for trigger words
def contains_trigger_word(text, trigger_words):
    for word in trigger_words:
        if word in text:
            return True
    return False

# Main loop with conversation history
conversation_history = [{'role': 'system', 'content': 'You are a helpful assistant that teaches user how to speak Japanese'}]

if __name__ == "__main__":
    while True:
        jap_or_nat = input("Would you like to speak in Japanese or your mothertone?j/m")
        if jap_or_nat == "m":
            print("Start recording, please speak Mandarin!")
            record_audio("audio/speech.wav")
            # Transcribe audio
            response_text = stt_model('audio/speech.wav','zh')
            # Process and respond to the speech with history
            response_from_gpt = gpt(response_text, conversation_history)
            tts_model(response_from_gpt['Japanese'])
        else:
            print("Start recording, please speak Mandarin!")
            # Record audio
            record_audio("audio/speech.wav")
            # Transcribe audio
            response_text = stt_model('audio/speech.wav','ja')
            # Process and respond to the speech with history
            response_from_gpt = gpt_convo(response_text, conversation_history)
            tts_model(response_from_gpt['Japanese'])

        while True:
            reply_or_not = input("Would you like to replay or continue? r/c")
            if reply_or_not == "r":
                playsound("audio/speech.mp3")
            else:
                break

