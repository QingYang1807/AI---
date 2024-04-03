import os
import streamlit as st
import torch
from pathlib import Path
from typing import Annotated, Union
from peft import AutoPeftModelForCausalLM, PeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    AutoModel,
)

# 使用方式，命令行中执行：streamlit run web_demo_gradio_stream.py

MODEL_PATH = '/root/.cache/modelscope/hub/ZhipuAI/chatglm3-6b'
TOKENIZER_PATH = '/root/.cache/modelscope/hub/ZhipuAI/chatglm3-6b'

ModelType = Union[PreTrainedModel, PeftModelForCausalLM]
TokenizerType = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

# MODEL_PATH = os.environ.get('MODEL_PATH', 'THUDM/chatglm3-6b')
# TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", MODEL_PATH)

st.set_page_config(
    page_title="ChatGLM3-6B Streamlit Simple Demo",
    page_icon=":robot:",
    layout="wide"
)

def _resolve_path(path: Union[str, Path]) -> Path:
    return Path(path).expanduser().resolve()
    
def load_model_and_tokenizer(model_dir: Union[str, Path]) -> tuple[ModelType, TokenizerType]:
    model_dir = _resolve_path(model_dir)
    if (model_dir / 'adapter_config.json').exists():
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=True, device_map='auto'
        )
        tokenizer_dir = model.peft_config['default'].base_model_name_or_path
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=True, device_map='auto'
        )
        tokenizer_dir = model_dir
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir, trust_remote_code=True
    )
    return model, tokenizer


@st.cache_resource
def get_model():
    model_dir = 'output/checkpoint-2000'
    model, tokenizer = load_model_and_tokenizer(model_dir)
    # tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    # model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True, device_map="auto").eval()
    return tokenizer, model


# 加载Chatglm3的model和tokenizer
tokenizer, model = get_model()


if "history" not in st.session_state:
    st.session_state.history = []
if "past_key_values" not in st.session_state:
    st.session_state.past_key_values = None

max_length = st.sidebar.slider("max_length", 0, 32768, 8192, step=1)
top_p = st.sidebar.slider("top_p", 0.0, 1.0, 0.8, step=0.01)
temperature = st.sidebar.slider("temperature", 0.0, 1.0, 0.6, step=0.01)

buttonClean = st.sidebar.button("清理会话历史", key="clean")
if buttonClean:
    st.session_state.history = []
    st.session_state.past_key_values = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    st.rerun()

for i, message in enumerate(st.session_state.history):
    if message["role"] == "user":
        with st.chat_message(name="user", avatar="user"):
            st.markdown(message["content"])
    else:
        with st.chat_message(name="assistant", avatar="assistant"):
            st.markdown(message["content"])

with st.chat_message(name="user", avatar="user"):
    input_placeholder = st.empty()
with st.chat_message(name="assistant", avatar="assistant"):
    message_placeholder = st.empty()

prompt_text = st.chat_input("请输入您的问题")
if prompt_text:
    input_placeholder.markdown(prompt_text)
    history = st.session_state.history
    past_key_values = st.session_state.past_key_values
    for response, history, past_key_values in model.stream_chat(
            tokenizer,
            prompt_text,
            history,
            past_key_values=past_key_values,
            max_length=max_length,
            top_p=top_p,
            temperature=temperature,
            return_past_key_values=True,
    ):
        message_placeholder.markdown(response)
    st.session_state.history = history
    st.session_state.past_key_values = past_key_values