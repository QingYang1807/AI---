# 整合前测试


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
import os
import platform
from transformers import AutoTokenizer, AutoModel

MODEL_PATH = '/root/.cache/modelscope/hub/ZhipuAI/chatglm3-6b'
TOKENIZER_PATH = '/root/.cache/modelscope/hub/ZhipuAI/chatglm3-6b'

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True, device_map="auto").eval()

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False

welcome_prompt = "欢迎使用 ChatGLM3-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序"
```


    Loading checkpoint shards:   0%|          | 0/7 [00:00<?, ?it/s]


    /root/miniconda3/lib/python3.10/site-packages/torch/_utils.py:831: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
      return self.fget.__get__(instance, owner)()



```python
def build_prompt(history):
    prompt = welcome_prompt
    for query, response in history:
        prompt += f"\n\n用户：{query}"
        prompt += f"\n\nChatGLM3-6B：{response}"
    return prompt
```


```python
past_key_values, history = None, []
global stop_stream
print(welcome_prompt)
while True:
    query = input("\n用户：")
    if query.strip() == "stop":
        break
    if query.strip() == "clear":
        past_key_values, history = None, []
        os.system(clear_command)
        print(welcome_prompt)
        continue
    print("\nChatGLM：", end="")
    current_length = 0
    for response, history, past_key_values in model.stream_chat(tokenizer, query, history=history, top_p=1,
                                                                temperature=0.01,
                                                                past_key_values=past_key_values,
                                                                return_past_key_values=True):
        if stop_stream:
            stop_stream = False
            break
        else:
            print(response[current_length:], end="", flush=True)
            current_length = len(response)
    print("")
```

    欢迎使用 ChatGLM3-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序


    
    用户： 你好


    
    ChatGLM：你好👋！我是人工智能助手 ChatGLM3-6B，很高兴见到你，欢迎问我任何问题。


    
    用户： 类型#裙*裙长#半身裙


    
    ChatGLM：你好！根据你的描述，我理解你想了解有关“裙”这个类型的信息，特别是“裙长”为“半身裙”的相关内容。下面是我为你准备的一些信息：
    
    半身裙是一种长度及膝以下的裙子，通常到膝盖附近。它是一种非常受欢迎的裙子类型，可以搭配多种上衣，并适用于各种场合。半身裙可以由各种材料制成，如棉、丝绸、牛仔布等。
    
    半身裙的款式多种多样，有包臀型、A字型、百褶型等。其中，包臀型半身裙适合曲线玲珑的身材，A字型半身裙则适合矩形身材，而百褶型半身裙则适合丰满的身材。
    
    半身裙可以搭配多种上衣，如T恤、衬衫、毛衣等。在搭配时，可以根据场合、个人喜好和身材特点进行选择。例如，在夏季，可以搭配轻薄舒适的T恤和凉鞋；在冬季，可以搭配厚实的毛衣和靴子。
    
    总之，半身裙是一种非常受欢迎的裙子类型，适合各种场合和身材特点。希望我的回答能对你有所帮助！



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    Cell In[6], line 5
          3 print(welcome_prompt)
          4 while True:
    ----> 5     query = input("\n用户：")
          6     if query.strip() == "stop":
          7         break


    File ~/miniconda3/lib/python3.10/site-packages/ipykernel/kernelbase.py:1262, in Kernel.raw_input(self, prompt)
       1260     msg = "raw_input was called, but this frontend does not support input requests."
       1261     raise StdinNotImplementedError(msg)
    -> 1262 return self._input_request(
       1263     str(prompt),
       1264     self._parent_ident["shell"],
       1265     self.get_parent("shell"),
       1266     password=False,
       1267 )


    File ~/miniconda3/lib/python3.10/site-packages/ipykernel/kernelbase.py:1305, in Kernel._input_request(self, prompt, ident, parent, password)
       1302 except KeyboardInterrupt:
       1303     # re-raise KeyboardInterrupt, to truncate traceback
       1304     msg = "Interrupted by user"
    -> 1305     raise KeyboardInterrupt(msg) from None
       1306 except Exception:
       1307     self.log.warning("Invalid Message:", exc_info=True)


    KeyboardInterrupt: Interrupted by user



```python
def main():
    past_key_values, history = None, []
    global stop_stream
    print(welcome_prompt)
    while True:
        query = input("\n用户：")
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            past_key_values, history = None, []
            os.system(clear_command)
            print(welcome_prompt)
            continue
        print("\nChatGLM：", end="")
        current_length = 0
        for response, history, past_key_values in model.stream_chat(tokenizer, query, history=history, top_p=1,
                                                                    temperature=0.01,
                                                                    past_key_values=past_key_values,
                                                                    return_past_key_values=True):
            if stop_stream:
                stop_stream = False
                break
            else:
                print(response[current_length:], end="", flush=True)
                current_length = len(response)
        print("")

```


```python
# 定义模型和分词器加载函数
# 这个函数负责从给定的目录加载模型和分词器。它首先将输入路径标准化，然后检查是否存在adapter_config.json文件来决定是加载标准的AutoModelForCausalLM模型还是特殊的AutoPeftModelForCausalLM模型。根据所加载的模型类型，它还决定从哪个目录加载分词器。
from pathlib import Path
from typing import Annotated, Union
from peft import AutoPeftModelForCausalLM, PeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

ModelType = Union[PreTrainedModel, PeftModelForCausalLM]
TokenizerType = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]


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

```


```python
def _resolve_path(path: Union[str, Path]) -> Path:
    return Path(path).expanduser().resolve()
```


```python
model_dir = 'output/checkpoint-2000'
```


```python
 model, tokenizer = load_model_and_tokenizer(model_dir)
```


    Loading checkpoint shards:   0%|          | 0/7 [00:00<?, ?it/s]



```python
prompt = '类型#裙*裙长#半身裙'
response, _ = model.chat(tokenizer, prompt)
# 最后，打印出响应。
print(response)
```

    这款半身裙采用柔软的纯棉面料，亲肤舒适，不刺激皮肤，穿着舒适。时尚的版型设计，穿起来很显瘦。时尚的荷叶边裙摆，凸显出女性柔美的气质。


# 整合后的cli_demo.py


```python
import os
import platform
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

model_dir = 'output/checkpoint-2000' # 输入微调后的Checkpoint目录地址
model, tokenizer = load_model_and_tokenizer(model_dir)


ModelType = Union[PreTrainedModel, PeftModelForCausalLM]
TokenizerType = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]


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

def main():
    past_key_values, history = None, []
    global stop_stream
    print(welcome_prompt)
    while True:
        query = input("\n用户：")
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            past_key_values, history = None, []
            os.system(clear_command)
            print(welcome_prompt)
            continue
        print("\nChatGLM：", end="")
        current_length = 0
        for response, history, past_key_values in model.stream_chat(tokenizer, query, history=history, top_p=1,
                                                                    temperature=0.01,
                                                                    past_key_values=past_key_values,
                                                                    return_past_key_values=True):
            if stop_stream:
                stop_stream = False
                break
            else:
                print(response[current_length:], end="", flush=True)
                current_length = len(response)
        print("")

```


    Loading checkpoint shards:   0%|          | 0/7 [00:00<?, ?it/s]



```python
main()
```

    欢迎使用 ChatGLM3-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序


    
    用户： 类型#裙*裙长#半身裙


    
    ChatGLM：这款半身裙采用优质面料，柔软舒适，穿着舒适，不透视，不透风，穿着舒适，不勒肉。裙身采用纯色设计，简约大方，百搭实穿。裙摆采用不规则设计，时尚有型，搭配起来更有层次感。


    
    用户： stop

