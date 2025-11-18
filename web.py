import streamlit as st
from audiorecorder import audiorecorder
from tempfile import NamedTemporaryFile
import librosa
import base64
from langchain_google_genai import ChatGoogleGenerativeAI
from io import BytesIO
import os
import scipy
import torchaudio as ta

from chatterbox.mtl_tts import (
    ChatterboxMultilingualTTS,
    SUPPORTED_LANGUAGES
)

import torch

original_torch_load = torch.load

def patched_torch_load(f, map_location=None, **kwargs):
    if map_location is None:
        map_location = 'cpu'
    return original_torch_load(f, map_location=map_location, **kwargs)

torch.load = patched_torch_load

@st.cache_resource  # 👈 Add the caching decorator
def load_model():
    try:
        map_location = torch.device('cpu')
        TTS_MODEL = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
        torch.load = original_torch_load
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    return TTS_MODEL

def embed_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
    html = f"""<audio controls>
    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    Your browser does not support the audio element.
</audio>"""
    return html

def tts_inference(target_text, target_audio, TTS_MODEL, lang):
    with NamedTemporaryFile(suffix=".mp3") as temp:
        file_name = temp.name
        tts_wav = TTS_MODEL.generate(
            target_text,
            audio_prompt_path = target_audio,
            language_id = lang
        )
        ta.save(f"{file_name}", tts_wav, TTS_MODEL.sr)
        tts_embed = embed_audio(file_name)
        return tts_embed
    
def clear_history():
    st.session_state.messages = []

def rewind():
    if st.session_state.messages:
        msg = st.session_state.messages.pop()
        while (msg.get('role', '') != 'user') and st.session_state.messages:
            msg = st.session_state.messages.pop()

def language_options(only_list=False):
    translate_dict = {
        v:k for k,v in SUPPORTED_LANGUAGES.items() if k != 'zh'
    }
    if only_list:
        return [i for i in translate_dict.keys()]
    else:
        return translate_dict
    
lang_list = language_options(only_list=True)

st.title("Voice Chatbot")

if "messages" not in st.session_state:
    clear_history()

with st.sidebar:
    st.header("Model")
    lang = st.selectbox("Language", lang_list)
    TTS_MODEL = load_model()
    st.header("Control")
    gemini_api_key = st.text_input("GEMINI API Key", key="chatbot_api_key", type="password")
    voice_embed = st.toggle('Show Audio', value=True)
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        st.button("Rewind", on_click=rewind, use_container_width=True, type='primary')
    with btn_col2:
        st.button("Clear", on_click=clear_history, use_container_width=True)

st.subheader("마이크 녹음 테스트")

audio = audiorecorder("녹음 시작", "녹음 정지")

if len(audio) > 0:
    st.success("녹음 완료")

    with NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
        tmp_path = tmp_file.name
        audio.export(tmp_path, format="mp3")
        st.write(f"저장된 파일: {tmp_path}")
