# CosyVoice2-0.5B MCP服务器

基于Model Context Protocol (MCP) 的CosyVoice2-0.5B语音合成服务器，提供远程语音合成功能。

## 功能特性

- **零样本语音合成** (Zero-shot TTS): 根据参考音频生成指定文本的语音
- **跨语言语音合成** (Cross-lingual TTS): 支持细粒度控制标记，如[laughter]
- **指令式语音合成** (Instruct TTS): 根据指令控制语音风格
- **音频文件管理**: 加载和保存音频文件
- **WebSocket协议**: 支持远程访问和实时通信

## 安装要求

### 1. 安装CosyVoice

```bash
# 克隆CosyVoice仓库
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice

# 创建conda环境
conda create -n cosyvoice python=3.10
conda activate cosyvoice
conda install -y -c conda-forge pynini==2.1.5
pip install -r requirements.txt

# 下载模型
from modelscope import snapshot_download
snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')
```

### 2. 安装MCP服务器依赖

```bash
pip install websockets asyncio
```

## 使用方法

### 1. 启动MCP服务器

```bash
# 基本启动
python cosyvoice_mcp_server.py

# 指定端口和预加载模型
python cosyvoice_mcp_server.py --port 8080 --model-dir pretrained_models/CosyVoice2-0.5B
```

### 2. 使用客户端

```bash
# 运行演示
python cosyvoice_mcp_client.py --demo

# 交互模式
python cosyvoice_mcp_client.py --interactive
```

## MCP工具说明

### initialize_model
初始化CosyVoice2-0.5B模型
- `model_dir`: 模型目录路径
- `load_jit`: 是否加载JIT模式
- `load_trt`: 是否加载TensorRT模式
- `fp16`: 是否使用FP16精度

### zero_shot_tts
零样本语音合成
- `text`: 要合成的文本
- `prompt_text`: 参考音频对应的文本
- `prompt_audio_base64`: 参考音频的base64编码
- `stream`: 是否使用流式生成

### cross_lingual_tts
跨语言语音合成
- `text`: 要合成的文本（支持[laughter]等标记）
- `prompt_audio_base64`: 参考音频的base64编码
- `stream`: 是否使用流式生成

### instruct_tts
指令式语音合成
- `text`: 要合成的文本
- `instruction`: 语音风格指令
- `prompt_audio_base64`: 参考音频的base64编码
- `stream`: 是否使用流式生成

### get_model_info
获取模型信息

### save_audio
保存base64编码的音频文件
- `audio_base64`: base64编码的音频数据
- `filepath`: 保存路径

### load_audio
加载音频文件并返回base64编码
- `filepath`: 音频文件路径

## API使用示例

### Python客户端示例

```python
import asyncio
import json
import websockets

async def example_usage():
    # 连接到服务器
    websocket = await websockets.connect("ws://localhost:8080")
    
    # 初始化模型
    request = {
        "action": "call_function",
        "name": "initialize_model",
        "arguments": {"model_dir": "/home/rh/Program/MCP_Tools/CosyVoice/pretrained_models/CosyVoice2-0.5B"}
    }
    await websocket.send(json.dumps(request))
    response = await websocket.recv()
    print("初始化结果:", json.loads(response))
    
    # 加载参考音频
    request = {
        "action": "call_function",
        "name": "load_audio",
        "arguments": {"filepath": "reference_audio.wav"}
    }
    await websocket.send(json.dumps(request))
    response = await websocket.recv()
    audio_data = json.loads(response)
    
    if audio_data["result"]["success"]:
        # 零样本语音合成
        request = {
            "action": "call_function",
            "name": "zero_shot_tts",
            "arguments": {
                "text": "你好，这是一个测试。",
                "prompt_text": "参考音频对应的文本",
                "prompt_audio_base64": audio_data["result"]["audio_base64"]
            }
        }
        await websocket.send(json.dumps(request))
        response = await websocket.recv()
        result = json.loads(response)
        
        # 保存合成的音频
        if result["result"]["success"]:
            for i, audio_result in enumerate(result["result"]["results"]):
                save_request = {
                    "action": "call_function",
                    "name": "save_audio",
                    "arguments": {
                        "audio_base64": audio_result["audio_base64"],
                        "filepath": f"output_{i}.wav"
                    }
                }
                await websocket.send(json.dumps(save_request))
                save_response = await websocket.recv()
                print(f"保存结果 {i}:", json.loads(save_response))
    
    await websocket.close()

# 运行示例
asyncio.run(example_usage())
```

## 注意事项

1. **音频格式**: 输入音频必须是16kHz的WAV格式
2. **内存使用**: 模型较大，确保有足够的GPU内存
3. **网络传输**: 音频数据通过base64编码传输，较大文件可能影响传输效率
4. **错误处理**: 所有API调用都包含错误处理，检查返回的`success`字段

## 故障排除

### 模型加载失败
- 检查模型路径是否正确
- 确保已安装所有依赖包
- 检查GPU内存是否足够

### 音频文件错误
- 确保音频文件存在且格式正确
- 检查文件权限
- 验证base64编码是否有效

### 网络连接问题
- 检查服务器是否正在运行
- 验证端口是否被占用
- 确认防火墙设置

## 扩展开发

服务器设计采用模块化架构，可以轻松添加新的语音合成功能：

1. 在`TOOLS`字典中添加新工具定义
2. 实现对应的处理函数
3. 更新客户端以支持新功能

## 许可证

请遵循CosyVoice项目的许可证要求。