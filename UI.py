import gradio as gr
import random
# chatbot引入
from interact import run_chatbot
# ASR 引入
from speech_paraformer.demo_online import asr

count = 0

def response(message, history):
    global count
    count += 1
    return run_chatbot(message, count)

def generateAudio(audio):
    return asr(audio)

def chat(audio):
    global count
    count += 1
    message = asr(audio)
    return message, run_chatbot(message, count)

if __name__ == '__main__':

    chatbot = gr.ChatInterface(fn=response)
    # audio_input = gr.Interface(
    #             fn=generateAudio, 
    #             inputs=gr.Audio(source="microphone", type="filepath"),  
    #             outputs="text"
    #             )
    
    audio_chatbot = gr.Interface(
                fn=chat, 
                inputs=gr.Audio(source="microphone", type="filepath"),  
                outputs=["text", "text"]
                )
    
    demo = gr.TabbedInterface([chatbot, audio_chatbot], ["聊天机器人", "语音聊天机器人"])
    # demo = gr.TabbedInterface([chatbot, audio_input], ["聊天机器人", "speech-to-text"])
    
    demo.launch(share=True)
