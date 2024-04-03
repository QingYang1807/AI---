# web_demo_gradio.py

# 整合前测试


```python
pip install gradio
```

    Looking in indexes: http://mirrors.aliyun.com/pypi/simple
    Collecting gradio
      Downloading http://mirrors.aliyun.com/pypi/packages/a7/06/29af2174e7aa1f3890e46483902bbb10628a90ddc053b2b02fcd201da89e/gradio-4.25.0-py3-none-any.whl (17.1 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m17.1/17.1 MB[0m [31m3.2 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hCollecting python-multipart>=0.0.9
      Downloading http://mirrors.aliyun.com/pypi/packages/3d/47/444768600d9e0ebc82f8e347775d24aef8f6348cf00e9fa0e81910814e6d/python_multipart-0.0.9-py3-none-any.whl (22 kB)
    Requirement already satisfied: typing-extensions~=4.0 in ./miniconda3/lib/python3.10/site-packages (from gradio) (4.9.0)
    Requirement already satisfied: pillow<11.0,>=8.0 in ./miniconda3/lib/python3.10/site-packages (from gradio) (10.2.0)
    Collecting ruff>=0.2.2
      Downloading http://mirrors.aliyun.com/pypi/packages/5a/8d/cc5d7078ffd2779a38a05bcff8ffaa24f524de51c46223a2dd2710a8c50e/ruff-0.3.5-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (8.7 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m8.7/8.7 MB[0m [31m3.3 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hRequirement already satisfied: jinja2<4.0 in ./miniconda3/lib/python3.10/site-packages (from gradio) (3.1.2)
    Collecting orjson~=3.0
      Downloading http://mirrors.aliyun.com/pypi/packages/fb/c9/d43cc61ebd197d6b50bcc2f6dad8d4102eaf1750e1d7e52d0b7fa8296f0f/orjson-3.10.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (144 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m144.8/144.8 kB[0m [31m3.2 MB/s[0m eta [36m0:00:00[0m00:01[0m
    [?25hRequirement already satisfied: pandas<3.0,>=1.0 in ./miniconda3/lib/python3.10/site-packages (from gradio) (2.2.1)
    Requirement already satisfied: pydantic>=2.0 in ./miniconda3/lib/python3.10/site-packages (from gradio) (2.6.4)
    Collecting tomlkit==0.12.0
      Downloading http://mirrors.aliyun.com/pypi/packages/68/4f/12207897848a653d03ebbf6775a29d949408ded5f99b2d87198bc5c93508/tomlkit-0.12.0-py3-none-any.whl (37 kB)
    Collecting aiofiles<24.0,>=22.0
      Downloading http://mirrors.aliyun.com/pypi/packages/c5/19/5af6804c4cc0fed83f47bff6e413a98a36618e7d40185cd36e69737f3b0e/aiofiles-23.2.1-py3-none-any.whl (15 kB)
    Collecting gradio-client==0.15.0
      Downloading http://mirrors.aliyun.com/pypi/packages/bc/63/534dfb5a0a3ff3ea6aa26dd394cef0e389f05bc15443fcb877b6a0cf2767/gradio_client-0.15.0-py3-none-any.whl (313 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m313.4/313.4 kB[0m [31m2.8 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hCollecting fastapi
      Downloading http://mirrors.aliyun.com/pypi/packages/c0/c1/2dc286475c8e2e455e431a1cf1cf29662c9f9290434161088ba039d77481/fastapi-0.110.1-py3-none-any.whl (91 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m91.9/91.9 kB[0m [31m2.9 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: matplotlib~=3.0 in ./miniconda3/lib/python3.10/site-packages (from gradio) (3.8.2)
    Collecting semantic-version~=2.0
      Downloading http://mirrors.aliyun.com/pypi/packages/6a/23/8146aad7d88f4fcb3a6218f41a60f6c2d4e3a72de72da1825dc7c8f7877c/semantic_version-2.10.0-py2.py3-none-any.whl (15 kB)
    Requirement already satisfied: huggingface-hub>=0.19.3 in ./miniconda3/lib/python3.10/site-packages (from gradio) (0.22.2)
    Requirement already satisfied: markupsafe~=2.0 in ./miniconda3/lib/python3.10/site-packages (from gradio) (2.1.3)
    Requirement already satisfied: packaging in ./miniconda3/lib/python3.10/site-packages (from gradio) (23.2)
    Collecting uvicorn>=0.14.0
      Downloading http://mirrors.aliyun.com/pypi/packages/73/f5/cbb16fcbe277c1e0b8b3ddd188f2df0e0947f545c49119b589643632d156/uvicorn-0.29.0-py3-none-any.whl (60 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m60.8/60.8 kB[0m [31m3.6 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: pyyaml<7.0,>=5.0 in ./miniconda3/lib/python3.10/site-packages (from gradio) (6.0.1)
    Collecting pydub
      Downloading http://mirrors.aliyun.com/pypi/packages/a6/53/d78dc063216e62fc55f6b2eebb447f6a4b0a59f55c8406376f76bf959b08/pydub-0.25.1-py2.py3-none-any.whl (32 kB)
    Collecting importlib-resources<7.0,>=1.3
      Downloading http://mirrors.aliyun.com/pypi/packages/75/06/4df55e1b7b112d183f65db9503bff189e97179b256e1ea450a3c365241e0/importlib_resources-6.4.0-py3-none-any.whl (38 kB)
    Requirement already satisfied: numpy~=1.0 in ./miniconda3/lib/python3.10/site-packages (from gradio) (1.26.3)
    Collecting ffmpy
      Downloading http://mirrors.aliyun.com/pypi/packages/1d/70/07914754979f5dd80bda947a0ffd181c08bfcb137b01c3c0cef45254d271/ffmpy-0.3.2.tar.gz (5.5 kB)
      Preparing metadata (setup.py) ... [?25ldone
    [?25hRequirement already satisfied: httpx>=0.24.1 in ./miniconda3/lib/python3.10/site-packages (from gradio) (0.27.0)
    Requirement already satisfied: typer[all]<1.0,>=0.9 in ./miniconda3/lib/python3.10/site-packages (from gradio) (0.12.0)
    Collecting altair<6.0,>=4.2.0
      Downloading http://mirrors.aliyun.com/pypi/packages/46/30/2118537233fa72c1d91a81f5908a7e843a6601ccc68b76838ebc4951505f/altair-5.3.0-py3-none-any.whl (857 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m857.8/857.8 kB[0m [31m3.4 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hCollecting websockets<12.0,>=10.0
      Downloading http://mirrors.aliyun.com/pypi/packages/58/0a/7570e15661a0a546c3a1152d95fe8c05480459bab36247f0acbf41f01a41/websockets-11.0.3-cp310-cp310-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (129 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m129.9/129.9 kB[0m [31m3.4 MB/s[0m eta [36m0:00:00[0ma [36m0:00:01[0m
    [?25hRequirement already satisfied: fsspec in ./miniconda3/lib/python3.10/site-packages (from gradio-client==0.15.0->gradio) (2023.12.2)
    Requirement already satisfied: toolz in ./miniconda3/lib/python3.10/site-packages (from altair<6.0,>=4.2.0->gradio) (0.12.0)
    Requirement already satisfied: jsonschema>=3.0 in ./miniconda3/lib/python3.10/site-packages (from altair<6.0,>=4.2.0->gradio) (4.20.0)
    Requirement already satisfied: httpcore==1.* in ./miniconda3/lib/python3.10/site-packages (from httpx>=0.24.1->gradio) (1.0.5)
    Requirement already satisfied: sniffio in ./miniconda3/lib/python3.10/site-packages (from httpx>=0.24.1->gradio) (1.3.0)
    Requirement already satisfied: anyio in ./miniconda3/lib/python3.10/site-packages (from httpx>=0.24.1->gradio) (4.2.0)
    Requirement already satisfied: certifi in ./miniconda3/lib/python3.10/site-packages (from httpx>=0.24.1->gradio) (2022.12.7)
    Requirement already satisfied: idna in ./miniconda3/lib/python3.10/site-packages (from httpx>=0.24.1->gradio) (3.4)
    Requirement already satisfied: h11<0.15,>=0.13 in ./miniconda3/lib/python3.10/site-packages (from httpcore==1.*->httpx>=0.24.1->gradio) (0.14.0)
    Requirement already satisfied: requests in ./miniconda3/lib/python3.10/site-packages (from huggingface-hub>=0.19.3->gradio) (2.31.0)
    Requirement already satisfied: tqdm>=4.42.1 in ./miniconda3/lib/python3.10/site-packages (from huggingface-hub>=0.19.3->gradio) (4.64.1)
    Requirement already satisfied: filelock in ./miniconda3/lib/python3.10/site-packages (from huggingface-hub>=0.19.3->gradio) (3.13.1)
    Requirement already satisfied: pyparsing>=2.3.1 in ./miniconda3/lib/python3.10/site-packages (from matplotlib~=3.0->gradio) (3.1.1)
    Requirement already satisfied: cycler>=0.10 in ./miniconda3/lib/python3.10/site-packages (from matplotlib~=3.0->gradio) (0.12.1)
    Requirement already satisfied: fonttools>=4.22.0 in ./miniconda3/lib/python3.10/site-packages (from matplotlib~=3.0->gradio) (4.47.0)
    Requirement already satisfied: kiwisolver>=1.3.1 in ./miniconda3/lib/python3.10/site-packages (from matplotlib~=3.0->gradio) (1.4.5)
    Requirement already satisfied: python-dateutil>=2.7 in ./miniconda3/lib/python3.10/site-packages (from matplotlib~=3.0->gradio) (2.8.2)
    Requirement already satisfied: contourpy>=1.0.1 in ./miniconda3/lib/python3.10/site-packages (from matplotlib~=3.0->gradio) (1.2.0)
    Requirement already satisfied: tzdata>=2022.7 in ./miniconda3/lib/python3.10/site-packages (from pandas<3.0,>=1.0->gradio) (2024.1)
    Requirement already satisfied: pytz>=2020.1 in ./miniconda3/lib/python3.10/site-packages (from pandas<3.0,>=1.0->gradio) (2024.1)
    Requirement already satisfied: annotated-types>=0.4.0 in ./miniconda3/lib/python3.10/site-packages (from pydantic>=2.0->gradio) (0.6.0)
    Requirement already satisfied: pydantic-core==2.16.3 in ./miniconda3/lib/python3.10/site-packages (from pydantic>=2.0->gradio) (2.16.3)
    [33mWARNING: typer 0.12.0 does not provide the extra 'all'[0m[33m
    [0mRequirement already satisfied: typer-slim[standard]==0.12.0 in ./miniconda3/lib/python3.10/site-packages (from typer[all]<1.0,>=0.9->gradio) (0.12.0)
    Requirement already satisfied: typer-cli==0.12.0 in ./miniconda3/lib/python3.10/site-packages (from typer[all]<1.0,>=0.9->gradio) (0.12.0)
    Requirement already satisfied: click>=8.0.0 in ./miniconda3/lib/python3.10/site-packages (from typer-slim[standard]==0.12.0->typer[all]<1.0,>=0.9->gradio) (8.1.7)
    Requirement already satisfied: rich>=10.11.0 in ./miniconda3/lib/python3.10/site-packages (from typer-slim[standard]==0.12.0->typer[all]<1.0,>=0.9->gradio) (13.7.1)
    Requirement already satisfied: shellingham>=1.3.0 in ./miniconda3/lib/python3.10/site-packages (from typer-slim[standard]==0.12.0->typer[all]<1.0,>=0.9->gradio) (1.5.4)
    Collecting starlette<0.38.0,>=0.37.2
      Downloading http://mirrors.aliyun.com/pypi/packages/fd/18/31fa32ed6c68ba66220204ef0be798c349d0a20c1901f9d4a794e08c76d8/starlette-0.37.2-py3-none-any.whl (71 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m71.9/71.9 kB[0m [31m3.4 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: attrs>=22.2.0 in ./miniconda3/lib/python3.10/site-packages (from jsonschema>=3.0->altair<6.0,>=4.2.0->gradio) (23.2.0)
    Requirement already satisfied: jsonschema-specifications>=2023.03.6 in ./miniconda3/lib/python3.10/site-packages (from jsonschema>=3.0->altair<6.0,>=4.2.0->gradio) (2023.12.1)
    Requirement already satisfied: referencing>=0.28.4 in ./miniconda3/lib/python3.10/site-packages (from jsonschema>=3.0->altair<6.0,>=4.2.0->gradio) (0.32.1)
    Requirement already satisfied: rpds-py>=0.7.1 in ./miniconda3/lib/python3.10/site-packages (from jsonschema>=3.0->altair<6.0,>=4.2.0->gradio) (0.16.2)
    Requirement already satisfied: six>=1.5 in ./miniconda3/lib/python3.10/site-packages (from python-dateutil>=2.7->matplotlib~=3.0->gradio) (1.16.0)
    Requirement already satisfied: exceptiongroup>=1.0.2 in ./miniconda3/lib/python3.10/site-packages (from anyio->httpx>=0.24.1->gradio) (1.2.0)
    Requirement already satisfied: urllib3<3,>=1.21.1 in ./miniconda3/lib/python3.10/site-packages (from requests->huggingface-hub>=0.19.3->gradio) (1.26.13)
    Requirement already satisfied: charset-normalizer<4,>=2 in ./miniconda3/lib/python3.10/site-packages (from requests->huggingface-hub>=0.19.3->gradio) (2.0.4)
    Requirement already satisfied: pygments<3.0.0,>=2.13.0 in ./miniconda3/lib/python3.10/site-packages (from rich>=10.11.0->typer-slim[standard]==0.12.0->typer[all]<1.0,>=0.9->gradio) (2.17.2)
    Requirement already satisfied: markdown-it-py>=2.2.0 in ./miniconda3/lib/python3.10/site-packages (from rich>=10.11.0->typer-slim[standard]==0.12.0->typer[all]<1.0,>=0.9->gradio) (3.0.0)
    Requirement already satisfied: mdurl~=0.1 in ./miniconda3/lib/python3.10/site-packages (from markdown-it-py>=2.2.0->rich>=10.11.0->typer-slim[standard]==0.12.0->typer[all]<1.0,>=0.9->gradio) (0.1.2)
    Building wheels for collected packages: ffmpy
      Building wheel for ffmpy (setup.py) ... [?25ldone
    [?25h  Created wheel for ffmpy: filename=ffmpy-0.3.2-py3-none-any.whl size=5583 sha256=68a8adb4ace2f8c79a4d6f82c4716dca2e6d31ab2ef085a3f39237fd2f978af4
      Stored in directory: /root/.cache/pip/wheels/f1/33/b9/967f1c6df43a4e6759a80bcc23e335ad8ea0d3c7be950bdcc7
    Successfully built ffmpy
    Installing collected packages: pydub, ffmpy, websockets, uvicorn, tomlkit, semantic-version, ruff, python-multipart, orjson, importlib-resources, aiofiles, starlette, gradio-client, fastapi, altair, gradio
    Successfully installed aiofiles-23.2.1 altair-5.3.0 fastapi-0.110.1 ffmpy-0.3.2 gradio-4.25.0 gradio-client-0.15.0 importlib-resources-6.4.0 orjson-3.10.0 pydub-0.25.1 python-multipart-0.0.9 ruff-0.3.5 semantic-version-2.10.0 starlette-0.37.2 tomlkit-0.12.0 uvicorn-0.29.0 websockets-11.0.3
    [33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m[33m
    [0mNote: you may need to restart the kernel to use updated packages.



```python
import os
import gradio as gr
import torch
from threading import Thread

from typing import Union, Annotated
from pathlib import Path
from peft import AutoPeftModelForCausalLM, PeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer
)
```


```python
ModelType = Union[PreTrainedModel, PeftModelForCausalLM]
TokenizerType = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
```


```python
MODEL_PATH = '/root/.cache/modelscope/hub/ZhipuAI/chatglm3-6b'
TOKENIZER_PATH = '/root/.cache/modelscope/hub/ZhipuAI/chatglm3-6b'
```


```python
def _resolve_path(path: Union[str, Path]) -> Path:
    return Path(path).expanduser().resolve()
```


```python
def load_model_and_tokenizer(
        model_dir: Union[str, Path], trust_remote_code: bool = True
) -> tuple[ModelType, TokenizerType]:
    model_dir = _resolve_path(model_dir)
    if (model_dir / 'adapter_config.json').exists():
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=trust_remote_code, device_map='auto'
        )
        tokenizer_dir = model.peft_config['default'].base_model_name_or_path
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=trust_remote_code, device_map='auto'
        )
        tokenizer_dir = model_dir
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir, trust_remote_code=trust_remote_code
    )
    return model, tokenizer
```


```python
model, tokenizer = load_model_and_tokenizer(MODEL_PATH, trust_remote_code=True)

```


    Loading checkpoint shards:   0%|          | 0/7 [00:00<?, ?it/s]


    /root/miniconda3/lib/python3.10/site-packages/torch/_utils.py:831: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
      return self.fget.__get__(instance, owner)()



```python
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [0, 2]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False
```


```python
def parse_text(text):
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text

```


```python
def predict(history, max_length, top_p, temperature):
    stop = StopOnTokens()
    messages = []
    for idx, (user_msg, model_msg) in enumerate(history):
        if idx == len(history) - 1 and not model_msg:
            messages.append({"role": "user", "content": user_msg})
            break
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if model_msg:
            messages.append({"role": "assistant", "content": model_msg})

    print("\n\n====conversation====\n", messages)
    model_inputs = tokenizer.apply_chat_template(messages,
                                                 add_generation_prompt=True,
                                                 tokenize=True,
                                                 return_tensors="pt").to(next(model.parameters()).device)
    streamer = TextIteratorStreamer(tokenizer, timeout=60, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = {
        "input_ids": model_inputs,
        "streamer": streamer,
        "max_new_tokens": max_length,
        "do_sample": True,
        "top_p": top_p,
        "temperature": temperature,
        "stopping_criteria": StoppingCriteriaList([stop]),
        "repetition_penalty": 1.2,
    }
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    for new_token in streamer:
        if new_token != '':
            history[-1][1] += new_token
            yield history

```


```python
with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">ChatGLM3-6B Gradio Simple Demo</h1>""")
    chatbot = gr.Chatbot()

    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10, container=False)
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit")
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")
            max_length = gr.Slider(0, 32768, value=8192, step=1.0, label="Maximum length", interactive=True)
            top_p = gr.Slider(0, 1, value=0.8, step=0.01, label="Top P", interactive=True)
            temperature = gr.Slider(0.01, 1, value=0.6, step=0.01, label="Temperature", interactive=True)


    def user(query, history):
        return "", history + [[parse_text(query), ""]]


    submitBtn.click(user, [user_input, chatbot], [user_input, chatbot], queue=False).then(
        predict, [chatbot, max_length, top_p, temperature], chatbot
    )
    emptyBtn.click(lambda: None, None, chatbot, queue=False)
```


```python
demo.queue()
```




    Gradio Blocks instance: 3 backend functions
    -------------------------------------------
    fn_index=0
     inputs:
     |-<gradio.components.textbox.Textbox object at 0x7fe6006160b0>
     |-<gradio.components.chatbot.Chatbot object at 0x7fe600658160>
     outputs:
     |-<gradio.components.textbox.Textbox object at 0x7fe6006160b0>
     |-<gradio.components.chatbot.Chatbot object at 0x7fe600658160>
    fn_index=1
     inputs:
     |-<gradio.components.chatbot.Chatbot object at 0x7fe600658160>
     |-<gradio.components.slider.Slider object at 0x7fe600617a90>
     |-<gradio.components.slider.Slider object at 0x7fe600616ec0>
     |-<gradio.components.slider.Slider object at 0x7fe600616ad0>
     outputs:
     |-<gradio.components.chatbot.Chatbot object at 0x7fe600658160>
    fn_index=2
     inputs:
     outputs:
     |-<gradio.components.chatbot.Chatbot object at 0x7fe600658160>




```python
demo.launch(server_name="127.0.0.1", server_port=7870, inbrowser=True, share=True)
```

    Running on local URL:  http://127.0.0.1:7870
    
    Could not create share link. Please check your internet connection or our status page: https://status.gradio.app.



<div><iframe src="http://127.0.0.1:7870/" width="100%" height="500" allow="autoplay; camera; microphone; clipboard-read; clipboard-write;" frameborder="0" allowfullscreen></iframe></div>





    



    
    No chat template is defined for this tokenizer - using a default chat template that implements the ChatML format (without BOS/EOS tokens!). If the default is not appropriate for your model, please set `tokenizer.chat_template` to an appropriate template. See https://huggingface.co/docs/transformers/main/chat_templating for more information.
    
    Exception in thread Thread-12 (generate):
    Traceback (most recent call last):
      File "/root/miniconda3/lib/python3.10/threading.py", line 1016, in _bootstrap_inner
        self.run()
      File "/root/miniconda3/lib/python3.10/threading.py", line 953, in run
        self._target(*self._args, **self._kwargs)
      File "/root/miniconda3/lib/python3.10/site-packages/torch/utils/_contextlib.py", line 115, in decorate_context
        return func(*args, **kwargs)
      File "/root/miniconda3/lib/python3.10/site-packages/transformers/generation/utils.py", line 1575, in generate
        result = self._sample(
      File "/root/miniconda3/lib/python3.10/site-packages/transformers/generation/utils.py", line 2697, in _sample
        outputs = self(
      File "/root/miniconda3/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
        return self._call_impl(*args, **kwargs)
      File "/root/miniconda3/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
        return forward_call(*args, **kwargs)
      File "/root/.cache/huggingface/modules/transformers_modules/chatglm3-6b/modeling_chatglm.py", line 937, in forward
        transformer_outputs = self.transformer(
      File "/root/miniconda3/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
        return self._call_impl(*args, **kwargs)
      File "/root/miniconda3/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
        return forward_call(*args, **kwargs)
      File "/root/.cache/huggingface/modules/transformers_modules/chatglm3-6b/modeling_chatglm.py", line 830, in forward
        hidden_states, presents, all_hidden_states, all_self_attentions = self.encoder(
      File "/root/miniconda3/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
        return self._call_impl(*args, **kwargs)
      File "/root/miniconda3/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
        return forward_call(*args, **kwargs)
      File "/root/.cache/huggingface/modules/transformers_modules/chatglm3-6b/modeling_chatglm.py", line 640, in forward
        layer_ret = layer(
      File "/root/miniconda3/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
        return self._call_impl(*args, **kwargs)
      File "/root/miniconda3/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
        return forward_call(*args, **kwargs)
      File "/root/.cache/huggingface/modules/transformers_modules/chatglm3-6b/modeling_chatglm.py", line 544, in forward
        attention_output, kv_cache = self.self_attention(
      File "/root/miniconda3/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
        return self._call_impl(*args, **kwargs)
      File "/root/miniconda3/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
        return forward_call(*args, **kwargs)
      File "/root/.cache/huggingface/modules/transformers_modules/chatglm3-6b/modeling_chatglm.py", line 376, in forward
        mixed_x_layer = self.query_key_value(hidden_states)
      File "/root/miniconda3/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
        return self._call_impl(*args, **kwargs)
      File "/root/miniconda3/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
        return forward_call(*args, **kwargs)
      File "/root/miniconda3/lib/python3.10/site-packages/torch/nn/modules/linear.py", line 114, in forward
        return F.linear(input, self.weight, self.bias)
    RuntimeError: "addmm_impl_cpu_" not implemented for 'Half'


    
    
    ====conversation====
     [{'role': 'user', 'content': '你好'}]


    Traceback (most recent call last):
      File "/root/miniconda3/lib/python3.10/site-packages/gradio/queueing.py", line 522, in process_events
        response = await route_utils.call_process_api(
      File "/root/miniconda3/lib/python3.10/site-packages/gradio/route_utils.py", line 260, in call_process_api
        output = await app.get_blocks().process_api(
      File "/root/miniconda3/lib/python3.10/site-packages/gradio/blocks.py", line 1741, in process_api
        result = await self.call_function(
      File "/root/miniconda3/lib/python3.10/site-packages/gradio/blocks.py", line 1308, in call_function
        prediction = await utils.async_iteration(iterator)
      File "/root/miniconda3/lib/python3.10/site-packages/gradio/utils.py", line 575, in async_iteration
        return await iterator.__anext__()
      File "/root/miniconda3/lib/python3.10/site-packages/gradio/utils.py", line 568, in __anext__
        return await anyio.to_thread.run_sync(
      File "/root/miniconda3/lib/python3.10/site-packages/anyio/to_thread.py", line 56, in run_sync
        return await get_async_backend().run_sync_in_worker_thread(
      File "/root/miniconda3/lib/python3.10/site-packages/anyio/_backends/_asyncio.py", line 2134, in run_sync_in_worker_thread
        return await future
      File "/root/miniconda3/lib/python3.10/site-packages/anyio/_backends/_asyncio.py", line 851, in run
        result = context.run(func, *args)
      File "/root/miniconda3/lib/python3.10/site-packages/gradio/utils.py", line 551, in run_sync_iterator_async
        return next(iterator)
      File "/root/miniconda3/lib/python3.10/site-packages/gradio/utils.py", line 734, in gen_wrapper
        response = next(iterator)
      File "/tmp/ipykernel_17087/781023984.py", line 32, in predict
        for new_token in streamer:
      File "/root/miniconda3/lib/python3.10/site-packages/transformers/generation/streamers.py", line 223, in __next__
        value = self.text_queue.get(timeout=self.timeout)
      File "/root/miniconda3/lib/python3.10/queue.py", line 179, in get
        raise Empty
    _queue.Empty



```python
!wget "https://cdn-media.huggingface.co/frpc-gradio-0.2/frpc_linux_amd64" -O /root/miniconda3/lib/python3.10/site-packages/gradio/frpc_linux_amd64_v0.2
```

    --2024-04-03 10:32:47--  https://cdn-media.huggingface.co/frpc-gradio-0.2/frpc_linux_amd64
    Resolving cdn-media.huggingface.co (cdn-media.huggingface.co)... 31.13.76.65, 2600:9000:2363:b400:19:6fb8:2ac0:93a1, 2600:9000:2363:7a00:19:6fb8:2ac0:93a1, ...
    Connecting to cdn-media.huggingface.co (cdn-media.huggingface.co)|31.13.76.65|:443... 


```python
# 下载失败，手动上传
```


```python
ll -h frpc_linux_amd64
```

    -rw-r--r-- 1 root 1.0M Apr  3 10:34 frpc_linux_amd64



```python
mv frpc_linux_amd64 frpc_linux_amd64_v0.2
```


```python
mv frpc_linux_amd64_v0.2 /root/miniconda3/lib/python3.10/site-packages/gradio
```

# 整合后的web_demo_gradio.py


```python
import os
import gradio as gr
import torch
from threading import Thread

from typing import Union, Annotated
from pathlib import Path
from peft import AutoPeftModelForCausalLM, PeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer
)

ModelType = Union[PreTrainedModel, PeftModelForCausalLM]
TokenizerType = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

# MODEL_PATH = '/root/.cache/modelscope/hub/ZhipuAI/chatglm3-6b'
# TOKENIZER_PATH = '/root/.cache/modelscope/hub/ZhipuAI/chatglm3-6b'

def _resolve_path(path: Union[str, Path]) -> Path:
    return Path(path).expanduser().resolve()


def load_model_and_tokenizer(
        model_dir: Union[str, Path], trust_remote_code: bool = True
) -> tuple[ModelType, TokenizerType]:
    model_dir = _resolve_path(model_dir)
    if (model_dir / 'adapter_config.json').exists():
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=trust_remote_code, device_map='auto'
        )
        tokenizer_dir = model.peft_config['default'].base_model_name_or_path
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=trust_remote_code, device_map='auto'
        )
        tokenizer_dir = model_dir
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir, trust_remote_code=trust_remote_code
    )
    return model, tokenizer

model_dir = 'output/checkpoint-2000' # 输入微调后的Checkpoint目录地址
model, tokenizer = load_model_and_tokenizer(model_dir, trust_remote_code=True)


class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [0, 2]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


def parse_text(text):
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text


def predict(history, max_length, top_p, temperature):
    stop = StopOnTokens()
    messages = []
    for idx, (user_msg, model_msg) in enumerate(history):
        if idx == len(history) - 1 and not model_msg:
            messages.append({"role": "user", "content": user_msg})
            break
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if model_msg:
            messages.append({"role": "assistant", "content": model_msg})

    print("\n\n====conversation====\n", messages)
    model_inputs = tokenizer.apply_chat_template(messages,
                                                 add_generation_prompt=True,
                                                 tokenize=True,
                                                 return_tensors="pt").to(next(model.parameters()).device)
    streamer = TextIteratorStreamer(tokenizer, timeout=60, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = {
        "input_ids": model_inputs,
        "streamer": streamer,
        "max_new_tokens": max_length,
        "do_sample": True,
        "top_p": top_p,
        "temperature": temperature,
        "stopping_criteria": StoppingCriteriaList([stop]),
        "repetition_penalty": 1.2,
    }
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    for new_token in streamer:
        if new_token != '':
            history[-1][1] += new_token
            yield history



with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">ChatGLM3-6B Gradio Simple Demo</h1>""")
    chatbot = gr.Chatbot()

    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10, container=False)
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit")
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")
            max_length = gr.Slider(0, 32768, value=8192, step=1.0, label="Maximum length", interactive=True)
            top_p = gr.Slider(0, 1, value=0.8, step=0.01, label="Top P", interactive=True)
            temperature = gr.Slider(0.01, 1, value=0.6, step=0.01, label="Temperature", interactive=True)


    def user(query, history):
        return "", history + [[parse_text(query), ""]]


    submitBtn.click(user, [user_input, chatbot], [user_input, chatbot], queue=False).then(
        predict, [chatbot, max_length, top_p, temperature], chatbot
    )
    emptyBtn.click(lambda: None, None, chatbot, queue=False)

demo.queue()
demo.launch(server_name="127.0.0.1", server_port=7870, inbrowser=True, share=True)
```


    Loading checkpoint shards:   0%|          | 0/7 [00:00<?, ?it/s]


    /root/miniconda3/lib/python3.10/site-packages/torch/_utils.py:831: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
      return self.fget.__get__(instance, owner)()


    Running on local URL:  http://127.0.0.1:7870
    
    Could not create share link. Please check your internet connection or our status page: https://status.gradio.app.



<div><iframe src="http://127.0.0.1:7870/" width="100%" height="500" allow="autoplay; camera; microphone; clipboard-read; clipboard-write;" frameborder="0" allowfullscreen></iframe></div>





    



    
    No chat template is defined for this tokenizer - using a default chat template that implements the ChatML format (without BOS/EOS tokens!). If the default is not appropriate for your model, please set `tokenizer.chat_template` to an appropriate template. See https://huggingface.co/docs/transformers/main/chat_templating for more information.
    


    
    
    ====conversation====
     [{'role': 'user', 'content': '你好'}]
    
    
    ====conversation====
     [{'role': 'user', 'content': '你好'}, {'role': 'assistant', 'content': '你好，我是人工智能助手。有什么我可以帮助你的吗？'}, {'role': 'user', 'content': '类型#裙*裙长#半身裙'}]

