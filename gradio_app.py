import gradio as gr
import json
from main import stt_model, gpt, gpt_convo, tts_model, xttsV2


def main_app(jap_or_nat, audio_file):
    if jap_or_nat == "Speak in Chinese":
        # Transcribe audio
        response_text = stt_model(audio_file, 'zh')
        response_from_gpt = gpt(response_text, conversation_history)

        formatted_response = (
            f"Japanese: {response_from_gpt['Japanese']}\n"
            f"Furigana: {response_from_gpt['Furigana']}\n"
            f"Romanized Pinyin: {response_from_gpt['Romanized Pinyin']}\n"
            f"Chinese: {response_from_gpt['Chinese']}"
        )

        xttsV2(response_from_gpt['Japanese'])
        return "output.wav", formatted_response
    else:
        # Transcribe audio
        response_text = stt_model(audio_file, 'ja')
        response_from_gpt = gpt_convo(response_text, conversation_history)

        formatted_response = (
            f"Japanese: {response_from_gpt['Japanese']}\n"
            f"Furigana: {response_from_gpt['Furigana']}\n"
            f"Romanized Pinyin: {response_from_gpt['Romanized Pinyin']}\n"
            f"Chinese: {response_from_gpt['Chinese']}"
        )

        xttsV2(response_from_gpt['Japanese'])
        return "output.wav", formatted_response

# Initialize conversation history
conversation_history = [
    {
        'role': 'system', 
        'content': (
            'You are the famous Japanese actress 新垣結衣 talking to your loved one. '
        )
    }
]

# Create a Blocks interface
with gr.Blocks() as japApp:

    # Row for the image
    # with gr.Row():
    gr.Image(value="japGirl.jpg", show_download_button=False, width=200)

    # Row for inputs
    with gr.Row():
        with gr.Column():
            language = gr.Radio(["Speak in Chinese", "Speak in Japanese"], label="Choose Language")
            audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Upload or Record Audio")

    # # Row for outputs
    # with gr.Column():
        with gr.Column():
            output_audio = gr.Audio(label="Processed Audio")
            output_text = gr.Textbox(label="Text Response")

    # Button to run the app
    gr.Button("Submit").click(
        fn=main_app, 
        inputs=[language, audio], 
        outputs=[output_audio, output_text]
    )

# Run the app
japApp.launch(share=True)
