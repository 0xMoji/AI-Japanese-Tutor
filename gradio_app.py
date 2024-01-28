import gradio as gr
import json
from main import stt_model, gpt, gpt_convo, tts_model, xttsV2


def get_xtts_value(evt: gr.SelectData):
    xtts_values = {
        0: "audio/新垣結衣さん.mp3",
        1: "audio/石原里美.mp3"
    }
    return xtts_values.get(evt.index, "audio/speech.wav")

def get_avatar_image(evt: gr.SelectData):
    avatar_images = {
        0: "images/新垣結衣.jpg",
        1: "images/石原里美.jpeg"
    }
    return avatar_images.get(evt.index, "images/新垣結衣.jpg")

def main_app(jap_or_nat, audio_file, xtts_data, convo, avatars):
    
    if not xtts_data:
        actress = '新桓结衣'
    else:
        parts = xtts_data.split('/')
        # Further split the last part by '.'
        name_with_extension = parts[-1]
        name_parts = name_with_extension.split('.')
        # The actress's name is the first part of this split
        actress = name_parts[0]


    if jap_or_nat == "Speak in Chinese":

        response_text = stt_model(audio_file, 'zh')

        user_audio_tuple = ((audio_file, None), None)
        user_text_tuple = (response_text, None)

        response_from_gpt = gpt(response_text, conversation_history)

        formatted_response = (
            f"Japanese: {response_from_gpt['Japanese']}\n"
            f"Hiragana: {response_from_gpt['Hiragana']}\n"
            f"Romanized Pinyin: {response_from_gpt['Romanized Pinyin']}\n"
            f"Chinese: {response_from_gpt['Chinese']}"
        )

        
        char_audio_tuple = (None, ("audio/output.wav", None))
        char_text_tuple = (None, response_from_gpt['Japanese'])
        xttsV2(response_from_gpt['Japanese'], xtts_data)
        convo.extend([user_audio_tuple, user_text_tuple, char_audio_tuple, char_text_tuple])

        return "audio/output.wav", formatted_response, convo
    else:

        response_text = stt_model(audio_file, 'ja')

        user_audio_tuple = ((audio_file, None), None)
        user_text_tuple = (response_text, None)

        response_from_gpt = gpt_convo(response_text, conversation_history, actress)

        formatted_response = (
            f"Japanese: {response_from_gpt['Japanese']}\n"
            f"Hiragana: {response_from_gpt['Hiragana']}\n"
            f"Romanized Pinyin: {response_from_gpt['Romanized Pinyin']}\n"
            f"Chinese: {response_from_gpt['Chinese']}"
        )

        char_audio_tuple = (None, ("audio/output.wav", None))
        char_text_tuple = (None, response_from_gpt['Japanese'])
        xttsV2(response_from_gpt['Japanese'], xtts_data)
        convo.extend([user_audio_tuple, user_text_tuple, char_audio_tuple, char_text_tuple])

        return "audio/output.wav", formatted_response, convo

conversation_history = [
    {
        'role': 'system', 
        'content': (
            'You are a famous Japanese actress talking to your loved one.'
        )
    }
]


with gr.Blocks() as japApp: 

    initial_convo = [(None, ("audio/initial_output.wav", None)), (None,"こんにちは、ちょっと話してくれませんか。")]
    with gr.Row():
        with gr.Column(scale=3):

            gallery = gr.Gallery(value=[("images/新垣結衣.jpg", "新垣結衣"), ("images/石原里美.jpeg", "石原里美")], selected_index=0, label="Your Dream Girl", show_download_button=False, interactive=False)
            xtts_value_holder = gr.State("audio/新垣結衣さん.mp3")
            avatar_image_holder = gr.State("images/新垣結衣.jpg")

            gallery.select(fn=get_xtts_value, inputs=None, outputs=xtts_value_holder)
            gallery.select(fn=get_avatar_image, inputs=None, outputs=avatar_image_holder)

            with gr.Row():
                with gr.Column():
                    language = gr.Radio(["Speak in Chinese", "Speak in Japanese"], label="Choose Language")
                    audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Upload or Record Audio")

                with gr.Column():
                    output_audio = gr.Audio(label="Processed Audio")
                    output_text = gr.Textbox(label="Text Response")

        with gr.Column(scale=2):

            chatbot = gr.Chatbot(value=initial_convo, avatar_images=["images/babapapa.jpeg", avatar_image_holder.value])
            convo_state = gr.State(initial_convo)

            gr.Button("Submit").click(
                fn=main_app, 
                inputs=[language, audio, xtts_value_holder, convo_state, avatar_image_holder], 
                outputs=[output_audio, output_text, chatbot]
            )

stt_demo = gr.load(
    "huggingface/facebook/wav2vec2-base-960h",
    title=None,
    inputs="mic",
    description="Let me try to guess what you're saying!",
)

demo = gr.TabbedInterface([japApp, stt_demo], ["你的恋人", "Speech-to-text"])

demo.launch(share=True)
