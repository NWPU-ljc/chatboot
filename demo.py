import gradio as gr
from speech_paraformer.demo_online import asr
from interact import run_chatbot

count = 0

def generateAudio(audio):
    return asr(audio)

def respond(message, chat_history):
        global count
        count += 1
        bot_message = run_chatbot(message, count)
        chat_history.append((message, bot_message))
        return "", chat_history

with gr.Blocks() as chat:
    chatbot = gr.Chatbot()
    msg = gr.Textbox() or 'text'
    clear = gr.ClearButton([msg, chatbot])
    msg.submit(respond, [msg, chatbot], [msg, chatbot])

audio_input = gr.Interface(
                fn=generateAudio, 
                inputs=gr.Audio(source="microphone", type="filepath"),  
                outputs="text"
                )

audio_chatbot = gr.Series(chat)

demo = gr.TabbedInterface([chat, audio_chatbot], ["聊天机器人", "语音聊天机器人"])

demo.launch()