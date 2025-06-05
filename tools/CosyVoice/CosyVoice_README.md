# 安装

## 克隆并安装

### 克隆仓库

``` bash
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
# If you failed to clone submodule due to network failures, please run following command until success
cd CosyVoice
git submodule update --init --recursive
```

### 安装 Conda: 请参阅 [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)

### 创建 Conda 环境

``` bash
conda create -n cosyvoice python=3.10
conda activate cosyvoice
# pynini is required by WeTextProcessing, use conda to install it as it can be executed on all platform.
conda install -y -c conda-forge pynini==2.1.5
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com

# If you encounter sox compatibility issues
# ubuntu
sudo apt-get install sox libsox-dev
# centos
sudo yum install sox sox-devel
```

### 模型下载

我们强烈建议您下载我们的预训练 CosyVoice2-0.5B CosyVoice-300M CosyVoice-300M-SFT CosyVoice-300M-Instruct 模型以及 CosyVoice-ttsfrd 资源。

``` python
# SDK模型下载
from modelscope import snapshot_download
snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')
snapshot_download('iic/CosyVoice-300M', local_dir='pretrained_models/CosyVoice-300M')
snapshot_download('iic/CosyVoice-300M-25Hz', local_dir='pretrained_models/CosyVoice-300M-25Hz')
snapshot_download('iic/CosyVoice-300M-SFT', local_dir='iic/CosyVoice-300M-SFT')
snapshot_download('iic/CosyVoice-300M-Instruct', local_dir='pretrained_models/CosyVoice-300M-Instruct')
snapshot_download('iic/CosyVoice-ttsfrd', local_dir='pretrained_models/CosyVoice-ttsfrd')
```

``` bash
# git模型下载，请确保已安装git lfs
mkdir -p pretrained_models
git clone https://www.modelscope.cn/iic/CosyVoice2-0.5B.git pretrained_models/CosyVoice2-0.5B
git clone https://www.modelscope.cn/iic/CosyVoice-300M.git pretrained_models/CosyVoice-300M
git clone https://www.modelscope.cn/iic/CosyVoice-300M-25Hz.git pretrained_models/CosyVoice-300M-25Hz
git clone https://www.modelscope.cn/iic/CosyVoice-300M-SFT.git pretrained_models/CosyVoice-300M-SFT
git clone https://www.modelscope.cn/iic/CosyVoice-300M-Instruct.git pretrained_models/CosyVoice-300M-Instruct
git clone https://www.modelscope.cn/iic/CosyVoice-ttsfrd.git pretrained_models/CosyVoice-ttsfrd
```

可选地，您可以解压 ttsfrd 资源并安装 ttsfrd 包以获得更好的文本规范化性能。

请注意，这一步不是必须的。如果您不安装 ttsfrd 包，我们将默认使用 WeTextProcessing。

``` bash
cd pretrained_models/CosyVoice-ttsfrd/
unzip resource.zip -d .
pip install ttsfrd_dependency-0.1-py3-none-any.whl
pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl
```

### 基本用法

我们强烈推荐使用 CosyVoice2-0.5B 以获得更好的性能。 请参考以下代码，了解每个模型的详细用法。

``` python
import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
```

#### CosyVoice2 用法

``` python
cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False)

# NOTE if you want to reproduce the results on https://funaudiollm.github.io/cosyvoice2, please add text_frontend=False during inference
# zero_shot usage
prompt_speech_16k = load_wav('zero_shot_prompt.wav', 16000)
for i, j in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
    torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# fine grained control, for supported control, check cosyvoice/tokenizer/tokenizer.py#L248
for i, j in enumerate(cosyvoice.inference_cross_lingual('在他讲述那个荒诞故事的过程中，他突然[laughter]停下来，因为他自己也被逗笑了[laughter]。', prompt_speech_16k, stream=False)):
    torchaudio.save('fine_grained_control_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# instruct usage
for i, j in enumerate(cosyvoice.inference_instruct2('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '用四川话说这句话', prompt_speech_16k, stream=False)):
    torchaudio.save('instruct_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
```

#### CosyVoice 用法

``` python
cosyvoice = CosyVoice('iic/CosyVoice-300M-SFT', load_jit=False, load_trt=False, fp16=False)
# sft usage
print(cosyvoice.list_available_spks())
# change stream=True for chunk stream inference
for i, j in enumerate(cosyvoice.inference_sft('你好，我是通义生成式语音大模型，请问有什么可以帮您的吗？', '中文女', stream=False)):
    torchaudio.save('sft_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M') # or change to pretrained_models/CosyVoice-300M-25Hz for 25Hz inference
# zero_shot usage, <|zh|><|en|><|jp|><|yue|><|ko|> for Chinese/English/Japanese/Cantonese/Korean
prompt_speech_16k = load_wav('zero_shot_prompt.wav', 16000)
for i, j in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
    torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
# cross_lingual usage
prompt_speech_16k = load_wav('cross_lingual_prompt.wav', 16000)
for i, j in enumerate(cosyvoice.inference_cross_lingual('<|en|>And then later on, fully acquiring that company. So keeping management in line, interest in line with the asset that\'s coming into the family is a reason why sometimes we don\'t buy the whole thing.', prompt_speech_16k, stream=False)):
    torchaudio.save('cross_lingual_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
# vc usage
prompt_speech_16k = load_wav('zero_shot_prompt.wav', 16000)
source_speech_16k = load_wav('cross_lingual_prompt.wav', 16000)
for i, j in enumerate(cosyvoice.inference_vc(source_speech_16k, prompt_speech_16k, stream=False)):
    torchaudio.save('vc_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M-Instruct')
# instruct usage, support <laughter></laughter><strong></strong>[laughter][breath]
for i, j in enumerate(cosyvoice.inference_instruct('在面对挑战时，他展现了非凡的<strong>勇气</strong>与<strong>智慧</strong>。', '中文男', 'Theo \'Crimson\', is a fiery, passionate rebel leader. Fights with fervor for justice, but struggles with impulsiveness.', stream=False)):
    torchaudio.save('instruct_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
```

### 启动网页演示

您可以使用我们的网页演示页面快速熟悉 CosyVoice。

详情请参见演示网站。

``` bash
# change iic/CosyVoice-300M-SFT for sft inference, or iic/CosyVoice-300M-Instruct for instruct inference
python3 webui.py --port 50000 --model_dir pretrained_models/CosyVoice-300M
```

### 高级用法

对于高级用户，我们在 examples/libritts/cosyvoice/run.sh 中提供了训练和推理脚本。

### 构建用于部署

如果您需要服务部署， 可以运行以下步骤。

``` bash
cd runtime/python
docker build -t cosyvoice:v1.0 .
# change iic/CosyVoice-300M to iic/CosyVoice-300M-Instruct if you want to use instruct inference
# for grpc usage
docker run -d --runtime=nvidia -p 50000:50000 cosyvoice:v1.0 /bin/bash -c "cd /opt/CosyVoice/CosyVoice/runtime/python/grpc && python3 server.py --port 50000 --max_conc 4 --model_dir iic/CosyVoice-300M && sleep infinity"
cd grpc && python3 client.py --port 50000 --mode <sft|zero_shot|cross_lingual|instruct>
# for fastapi usage
docker run -d --runtime=nvidia -p 50000:50000 cosyvoice:v1.0 /bin/bash -c "cd /opt/CosyVoice/CosyVoice/runtime/python/fastapi && python3 server.py --port 50000 --model_dir iic/CosyVoice-300M && sleep infinity"
cd fastapi && python3 client.py --port 50000 --mode <sft|zero_shot|cross_lingual|instruct>
```
