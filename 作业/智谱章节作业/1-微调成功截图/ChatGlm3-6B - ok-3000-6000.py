#!/usr/bin/env python
# coding: utf-8

# # 安装依赖

# In[1]:


get_ipython().system('pip install jieba>=0.42.1')
get_ipython().system('pip install ruamel_yaml>=0.18.6')
get_ipython().system('pip install rouge_chinese>=1.0.3')
get_ipython().system('pip install jupyter>=1.0.0')
get_ipython().system('pip install datasets>=2.17.1')
get_ipython().system('pip install peft>=0.10.0')
get_ipython().system('pip install transformers>=4.38.1')
get_ipython().system('pip install deepspeed==0.13.1')
get_ipython().system('pip install mpi4py>=3.1.5')
get_ipython().system('pip install typer')
get_ipython().system('pip install nltk')
get_ipython().system('pip install sentencepiece')


# # 准备训练数据集

# In[2]:


mkdir data


# In[3]:


import tarfile

# Specify the path to your tar.gz file and the output directory
tar_path = 'data/AdvertiseGen.tar.gz'
output_dir = './data/'

# Open the tar.gz file
with tarfile.open(tar_path, 'r:gz') as tar:
    tar.extractall(path=output_dir)


# In[4]:


ll data/AdvertiseGen


# # 查看样例数据

# In[5]:


import json
import random

# 加载JSON文件
file_path = 'data/AdvertiseGen/dev.json'  # JSON文件的路径

data = []
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        try:
            data.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")

# 检查数据长度，如果小于10，则取全部数据
sample_size = min(10, len(data))

# 随机选择10条或者全部数据（如果数据少于10条）
sample_data = random.sample(data, sample_size)

# 打印选中的记录
for item in sample_data:
    print(item)


# In[6]:


import json
import random

# 加载JSON文件
file_path = 'data/AdvertiseGen/train.json'  # JSON文件的路径

data = []
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        try:
            data.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")

# 检查数据长度，如果小于10，则取全部数据
sample_size = min(10, len(data))

# 随机选择10条或者全部数据（如果数据少于10条）
sample_data = random.sample(data, sample_size)

# 打印选中的记录
for item in sample_data:
    print(item)


# # lora微调

# ## 1. 准备数据集
# 我们使用 AdvertiseGen 数据集来进行微调。从 [Google Drive](https://drive.google.com/file/d/13_vf0xRTQsyneRKdD1bZIr93vBGOczrk/view?usp=sharing) 或者 [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/b3f119a008264b1cabd1/?dl=1) 下载处理好的 AdvertiseGen 数据集，将解压后的 AdvertiseGen 目录放到本目录的 `/data/` 下, 例如。
# > /media/zr/Data/Code/ChatGLM3/finetune_demo/data/AdvertiseGen
# 
# 接着，运行本代码来切割数据集

# In[7]:


import json
from typing import Union
from pathlib import Path


def _resolve_path(path: Union[str, Path]) -> Path:
    return Path(path).expanduser().resolve()


def _mkdir(dir_name: Union[str, Path]):
    dir_name = _resolve_path(dir_name)
    if not dir_name.is_dir():
        dir_name.mkdir(parents=True, exist_ok=False)


def convert_adgen(data_dir: Union[str, Path], save_dir: Union[str, Path]):
    def _convert(in_file: Path, out_file: Path):
        _mkdir(out_file.parent)
        with open(in_file, encoding='utf-8') as fin:
            with open(out_file, 'wt', encoding='utf-8') as fout:
                for line in fin:
                    dct = json.loads(line)
                    sample = {'conversations': [{'role': 'user', 'content': dct['content']},
                                                {'role': 'assistant', 'content': dct['summary']}]}
                    fout.write(json.dumps(sample, ensure_ascii=False) + '\n')

    data_dir = _resolve_path(data_dir)
    save_dir = _resolve_path(save_dir)

    train_file = data_dir / 'train.json'
    if train_file.is_file():
        out_file = save_dir / train_file.relative_to(data_dir)
        _convert(train_file, out_file)

    dev_file = data_dir / 'dev.json'
    if dev_file.is_file():
        out_file = save_dir / dev_file.relative_to(data_dir)
        _convert(dev_file, out_file)


convert_adgen('data/AdvertiseGen', 'data/AdvertiseGen_fix')


# ## 2. 使用命令行开始微调,我们使用 lora 进行微调
# 接着，我们仅需要将配置好的参数以命令行的形式传参给程序，就可以使用命令行进行高效微调，这里将 `/media/zr/Data/Code/ChatGLM3/venv/bin/python3` 换成你的 python3 的绝对路径以保证正常运行。

# ### 使用Model Scope库下载模型

# In[3]:


pip install modelscope


# In[5]:


from modelscope.models import Model
model = Model.from_pretrained('ZhipuAI/chatglm3-6b')


# ### 使用Hugging Face库下载模型(无法连接，下载失败)

# In[3]:


from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True).half().cuda()


# ## 找到模型下载目录

# In[11]:


ll /root/.cache/modelscope/hub/ZhipuAI/chatglm3-6b


# ## 找到Python3位置

# In[14]:


get_ipython().system('which python3')


# ## 开始微调-第一次微调

# In[1]:


cat configs/lora.yaml


# In[ ]:


get_ipython().system('/root/miniconda3/bin/python3 finetune_hf.py  data/AdvertiseGen_fix  /root/.cache/modelscope/hub/ZhipuAI/chatglm3-6b  configs/lora.yaml')


# In[ ]:


# 临时关闭了电脑，中断了输出，但是训练完成了


# ## 第二次微调-从保存点进行微调

# In[2]:


cat configs/lora.yaml


# In[6]:


get_ipython().system('/root/miniconda3/bin/python3 finetune_hf.py  data/AdvertiseGen_fix  /root/.cache/modelscope/hub/ZhipuAI/chatglm3-6b  configs/lora.yaml yes')


# In[ ]:


get_ipython().system('/root/miniconda3/bin/python3 finetune_hf.py  data/AdvertiseGen_fix  /root/.cache/modelscope/hub/ZhipuAI/chatglm3-6b  configs/lora.yaml yes')


# In[13]:


ll -h output/


# In[8]:


ll -a output/checkpoint-2000


# In[4]:


rm -rf output/.ipynb_checkpoints/


# In[5]:


ll output


# In[10]:


ll output/checkpoint-2000


# # 备份Checkpoint

# In[6]:


get_ipython().system('zip -r output-3000.zip output/')


# # 使用微调后的模型

# ## 在 inference_hf.py 中验证微调后的模型
# 
# 您可以在 `finetune_demo/inference_hf.py` 中使用我们的微调后的模型，仅需要一行代码就能简单的进行测试。

# In[1]:


cat output/checkpoint-2000/adapter_config.json


# In[7]:


get_ipython().system('python inference_hf.py /root/.cache/modelscope/hub/ZhipuAI/chatglm3-6b --prompt 类型#裙*裙长#半身裙')


# In[8]:


get_ipython().system('python inference_hf.py output/checkpoint-2000/ --prompt 类型#裙*裙长#半身裙')


# In[9]:


get_ipython().system('python inference_hf.py output/checkpoint-2000/ --prompt 类型#裙*裙长#半身裙')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




