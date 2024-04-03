# cli_demo_bad_word_ids.py

# 整合前测试


```python
import os
import platform

from transformers import AutoTokenizer, AutoModel
```


```python
ll /root/.cache/modelscope/hub/ZhipuAI/chatglm3-6b
```

    total 12195720
    -rw------- 1 root       4133 Apr  3 01:14 MODEL_LICENSE
    -rw------- 1 root       4478 Apr  3 01:19 README.md
    -rw------- 1 root       1317 Apr  3 01:14 config.json
    -rw------- 1 root         37 Apr  3 01:14 configuration.json
    -rw------- 1 root       2332 Apr  3 01:14 configuration_chatglm.py
    -rw------- 1 root      55596 Apr  3 01:14 modeling_chatglm.py
    -rw------- 1 root 1827781090 Apr  3 01:15 pytorch_model-00001-of-00007.bin
    -rw------- 1 root 1968299480 Apr  3 01:16 pytorch_model-00002-of-00007.bin
    -rw------- 1 root 1927415036 Apr  3 01:17 pytorch_model-00003-of-00007.bin
    -rw------- 1 root 1815225998 Apr  3 01:17 pytorch_model-00004-of-00007.bin
    -rw------- 1 root 1968299544 Apr  3 01:18 pytorch_model-00005-of-00007.bin
    -rw------- 1 root 1927415036 Apr  3 01:19 pytorch_model-00006-of-00007.bin
    -rw------- 1 root 1052808542 Apr  3 01:19 pytorch_model-00007-of-00007.bin
    -rw------- 1 root      20437 Apr  3 01:19 pytorch_model.bin.index.json
    -rw------- 1 root      14692 Apr  3 01:19 quantization.py
    -rw------- 1 root      11279 Apr  3 01:19 tokenization_chatglm.py
    -rw------- 1 root    1018370 Apr  3 01:19 tokenizer.model
    -rw------- 1 root        244 Apr  3 01:19 tokenizer_config.json



```python
MODEL_PATH = '/root/.cache/modelscope/hub/ZhipuAI/chatglm3-6b'
TOKENIZER_PATH = '/root/.cache/modelscope/hub/ZhipuAI/chatglm3-6b'
```


```python
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True, device_map="auto").eval()
```


    Loading checkpoint shards:   0%|          | 0/7 [00:00<?, ?it/s]


    /root/miniconda3/lib/python3.10/site-packages/torch/_utils.py:831: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
      return self.fget.__get__(instance, owner)()
    WARNING:root:Some parameters are on the meta device device because they were offloaded to the cpu.



```python
os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False

welcome_prompt = "欢迎使用 ChatGLM3-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序"
```


```python
# probability tensor contains either `inf`, `nan` or element < 0

bad_words = ["你好", "ChatGLM"]
bad_word_ids = [tokenizer.encode(bad_word, add_special_tokens=False) for bad_word in bad_words]
```


```python
def build_prompt(history):
    prompt = welcome_prompt
    for query, response in history:
        prompt += f"\n\n用户：{query}"
        prompt += f"\n\nChatGLM3-6B：{response}"
    return prompt
```


```python
def main():
    past_key_values, history = None, []
    global stop_stream
    print(welcome_prompt)
    while True:
        query = input("\n用户：")
        if query.strip().lower() == "stop":
            break
        if query.strip().lower() == "clear":
            past_key_values, history = None, []
            os.system(clear_command)
            print(welcome_prompt)
            continue

        # Attempt to generate a response
        try:
            print("\nChatGLM：", end="")
            current_length = 0
            response_generated = False
            for response, history, past_key_values in model.stream_chat(
                tokenizer, query, history=history, top_p=1,
                temperature=0.01,
                past_key_values=past_key_values,
                return_past_key_values=True,
                bad_words_ids=bad_word_ids  # assuming this is implemented correctly
            ):
                response_generated = True
                # Check if the response contains any bad words
                if any(bad_word in response for bad_word in bad_words):
                    print("我的回答涉嫌了 bad word")
                    break  # Break the loop if a bad word is detected

                # Otherwise, print the generated response
                print(response[current_length:], end="", flush=True)
                current_length = len(response)
            if not response_generated:
                print("没有生成任何回答。")
        except RuntimeError as e:
            print(f"生成文本时发生错误：{e}，这可能是涉及到设定的敏感词汇")

        print("")
```


```python
main()
```

    欢迎使用 ChatGLM3-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序


    
    用户： 什么是badword,列举一些，我用来测试


    
    ChatGLM："badword"是指在自然语言处理中，被认为是不合适或具有负面含义的词汇。这些词汇可能会被用于垃圾邮件、网络聊天、文本分析等场景中，可能会对其他用户造成负面影响。以下是一些常见的badword示例：
    
    1. 垃圾邮件中的常用badword：例如 "free", "cash", "prize", "gift", "money", "offer", "win", "free money", "credit card", "loan" 等。
    2. 社交媒体上的badword：例如 "spam", "scam", "fake", "scandal", "hacker", "cyber attack", "phishing", " identity theft", "data breach" 等。
    3. 在文本分析中常见的badword：例如 "the", "and", "a", "an", "in", "to", "of", "on", "that", "this", "for", "with", "as" 等。
    
    请注意，不同场景中的badword可能会有所不同，而且有些词汇在某些情况下可能并不被认为是badword。因此，在测试时，最好根据具体场景和目的来确定测试的badword列表。


    
    用户： stop



```python
main()
```

    欢迎使用 ChatGLM3-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序


    
    用户： 你好


    
    ChatGLM：生成文本时发生错误：probability tensor contains either `inf`, `nan` or element < 0，这可能是涉及到设定的敏感词汇
    


    
    用户： ChatGLM


    
    ChatGLM：生成文本时发生错误：probability tensor contains either `inf`, `nan` or element < 0，这可能是涉及到设定的敏感词汇
    


    
    用户： stop



```python

```


```python

```


```python
# 整合后的cli_demo_bad_word_ids.py
```


```python
import os
import platform
from pathlib import Path
from typing import Annotated, Union, Optional
from peft import AutoPeftModelForCausalLM, PeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    AutoModel,
    LogitsProcessorList,
)

ModelType = Union[PreTrainedModel, PeftModelForCausalLM]
TokenizerType = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False

welcome_prompt = "欢迎使用 ChatGLM3-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序"


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


def build_prompt(history):
    prompt = welcome_prompt
    for query, response in history:
        prompt += f"\n\n用户：{query}"
        prompt += f"\n\nChatGLM3-6B：{response}"
    return prompt


def main():
    model_dir = 'output/checkpoint-2000' # 输入微调后的Checkpoint目录地址
    model, tokenizer = load_model_and_tokenizer(model_dir)

    past_key_values, history = None, []
    global stop_stream
    print(welcome_prompt)
    while True:
        query = input("\n用户：")
        if query.strip().lower() == "stop":
            break
        if query.strip().lower() == "clear":
            past_key_values, history = None, []
            os.system(clear_command)
            print(welcome_prompt)
            continue

        # Attempt to generate a response
        try:
            print("\nChatGLM：", end="")
            current_length = 0
            response_generated = False
            for response, history, past_key_values in model.stream_chat(
                tokenizer, query, history=history, top_p=1,
                temperature=0.01,
                past_key_values=past_key_values,
                return_past_key_values=True,
                bad_words_ids=bad_word_ids  # assuming this is implemented correctly
            ):
                response_generated = True
                # Check if the response contains any bad words
                if any(bad_word in response for bad_word in bad_words):
                    print("我的回答涉嫌了 bad word")
                    break  # Break the loop if a bad word is detected

                # Otherwise, print the generated response
                print(response[current_length:], end="", flush=True)
                current_length = len(response)
            if not response_generated:
                print("没有生成任何回答。")
        except RuntimeError as e:
            print(f"生成文本时发生错误：{e}，这可能是涉及到设定的敏感词汇")

        print("")
```


```python
main()
```


    Loading checkpoint shards:   0%|          | 0/7 [00:00<?, ?it/s]


    WARNING:root:Some parameters are on the meta device device because they were offloaded to the cpu.
    WARNING:root:Some parameters are on the meta device device because they were offloaded to the cpu.


    欢迎使用 ChatGLM3-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序


    
    用户： 你好


    
    ChatGLM：生成文本时发生错误：probability tensor contains either `inf`, `nan` or element < 0，这可能是涉及到设定的敏感词汇
    


    
    用户： ChatGLM


    
    ChatGLM：生成文本时发生错误：probability tensor contains either `inf`, `nan` or element < 0，这可能是涉及到设定的敏感词汇
    


    
    用户： 类型#裙*裙长#半身裙


    
    ChatGLM：这款半身裙的版型设计非常合身，能够完美地展现出你的身材曲线，让你在穿着上更加有型有范。而裙身的设计更是别具一格，采用了一款精致的腰带，能够起到很好的收腰效果，让你穿着上更加有型。而裙身的设计更是别具一格，采用了一款精致的腰带，
